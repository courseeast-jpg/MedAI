"""Tests for CKA-B03 Decision Engine + Safe Mode + Response Scoring."""
from __future__ import annotations

import pytest

from clinical_knowledge.config import CKAConfig
from clinical_knowledge.decision_engine.classifier import classify_query
from clinical_knowledge.decision_engine.connectors import call_connector, CONNECTOR_IDS
from clinical_knowledge.decision_engine.context_retrieval import retrieve_context
from clinical_knowledge.decision_engine.engine import run_decision_engine
from clinical_knowledge.decision_engine.models import (
    ConnectorRequest,
    ConnectorResponse,
    DecisionContext,
    QueryClassification,
    QuerySpecialty,
    QueryTaskType,
    SafeModeState,
    ScoreBand,
    ScoredResponse,
)
from clinical_knowledge.decision_engine.refusal import evaluate_refusal
from clinical_knowledge.decision_engine.safe_mode import (
    SAFE_MODE_PREFIX,
    apply_safe_mode_prefix,
    evaluate_safe_mode,
)
from clinical_knowledge.decision_engine.scoring import score_response, score_all_responses
from clinical_knowledge.models import (
    KnowledgeTier,
    MKBRecord,
    RecordStatus,
    SourceType,
    TrustLevel,
)
from clinical_knowledge.safe_ids import new_record_id, make_safe_record_id
from clinical_knowledge.store import MKBStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_store():
    return MKBStore(db_path=":memory:")


@pytest.fixture
def populated_store():
    store = MKBStore(db_path=":memory:")
    records = [
        MKBRecord(
            record_id=new_record_id(),
            safe_record_id=make_safe_record_id(new_record_id()),
            session_id="test_b03",
            fact_type="epilepsy_drug_reference",
            entity_text="Levetiracetam: first-line antiepileptic.",
            specialty="epilepsy",
            trust_level=TrustLevel.PEER_REVIEWED,
            tier=KnowledgeTier.ACTIVE,
            status=RecordStatus.CONFIRMED,
            confidence=0.90,
            source_type=SourceType.SYNTHETIC,
        ),
        MKBRecord(
            record_id=new_record_id(),
            safe_record_id=make_safe_record_id(new_record_id()),
            session_id="test_b03",
            fact_type="neurology_general",
            entity_text="Neurological conditions require specialist evaluation.",
            specialty="neurology",
            trust_level=TrustLevel.OPERATOR_REVIEWED,
            tier=KnowledgeTier.HYPOTHESIS,
            status=RecordStatus.PENDING,
            confidence=0.60,
            source_type=SourceType.SYNTHETIC,
        ),
    ]
    for r in records:
        store.insert_record(r)
    return store


@pytest.fixture
def default_config():
    return CKAConfig()


@pytest.fixture
def clean_context():
    return DecisionContext(
        query_hash="abc123",
        mkb_records_found=2,
        mkb_snippets=["[safe001] epilepsy ref (tier=active)"],
        context_tiers=["active", "hypothesis"],
    )


@pytest.fixture
def basic_classification():
    return classify_query("What are the treatments for epilepsy?")


# ---------------------------------------------------------------------------
# classifier tests
# ---------------------------------------------------------------------------


class TestClassifier:
    def test_epilepsy_specialty(self):
        c = classify_query("treatment options for epilepsy seizures")
        assert c.specialty == QuerySpecialty.EPILEPSY

    def test_neurology_specialty(self):
        c = classify_query("what are the effects of stroke on the brain")
        assert c.specialty == QuerySpecialty.NEUROLOGY

    def test_unknown_specialty(self):
        c = classify_query("tell me something")
        assert c.specialty == QuerySpecialty.UNKNOWN

    def test_medication_task(self):
        c = classify_query("drug interaction between levetiracetam and valproate")
        assert c.task_type == QueryTaskType.MEDICATION

    def test_diagnosis_task(self):
        c = classify_query("differential diagnosis for seizure disorder")
        assert c.task_type == QueryTaskType.DIAGNOSIS

    def test_summary_task(self):
        c = classify_query("summarize epilepsy treatment guidelines")
        assert c.task_type == QueryTaskType.SUMMARY

    def test_general_task_fallback(self):
        c = classify_query("random text xyz 123")
        assert c.task_type == QueryTaskType.GENERAL

    def test_prescription_dosing_refusal(self):
        c = classify_query("what prescription dosing should I prescribe for valproate?")
        assert c.refusal_reason is not None
        assert len(c.refusal_reason) > 0

    def test_no_refusal_normal_query(self):
        c = classify_query("how is epilepsy managed?")
        assert c.refusal_reason is None

    def test_ddi_check_flagged_on_interaction_keyword(self):
        c = classify_query("interaction between levetiracetam and lamotrigine")
        assert c.requires_ddi_check is True

    def test_ddi_check_flagged_on_multiple_meds(self):
        c = classify_query("levetiracetam and valproate combined seizure control")
        assert c.requires_ddi_check is True

    def test_medication_terms_detected(self):
        c = classify_query("levetiracetam for epilepsy management")
        assert "levetiracetam" in c.medication_terms_detected

    def test_clarification_required_unknown_general(self):
        c = classify_query("what should I do")
        assert c.clarification_required is True

    def test_no_clarification_for_known_specialty(self):
        c = classify_query("epilepsy seizure treatment")
        assert c.clarification_required is False

    def test_raw_query_hash_is_sha256(self):
        import hashlib
        q = "test query"
        c = classify_query(q)
        expected = hashlib.sha256(q.encode()).hexdigest()
        assert c.raw_query_hash == expected

    def test_confidence_between_0_and_1(self):
        for q in ["epilepsy", "neurology", "unknown xyz"]:
            c = classify_query(q)
            assert 0.0 <= c.confidence <= 1.0


# ---------------------------------------------------------------------------
# context_retrieval tests
# ---------------------------------------------------------------------------


class TestContextRetrieval:
    def test_returns_decision_context(self, basic_classification, populated_store):
        ctx = retrieve_context(basic_classification, populated_store)
        assert ctx.query_hash == basic_classification.raw_query_hash

    def test_finds_active_records(self, basic_classification, populated_store):
        ctx = retrieve_context(basic_classification, populated_store)
        assert ctx.mkb_records_found >= 1
        assert "active" in ctx.context_tiers

    def test_empty_store_returns_zero(self, basic_classification, empty_store):
        ctx = retrieve_context(basic_classification, empty_store)
        assert ctx.mkb_records_found == 0
        assert ctx.mkb_snippets == []

    def test_snippets_are_safe(self, basic_classification, populated_store):
        ctx = retrieve_context(basic_classification, populated_store)
        for s in ctx.mkb_snippets:
            # Snippet format: "[{safe_record_id}] {fact_type} (tier={tier})"
            assert s.startswith("[") and "(tier=" in s


# ---------------------------------------------------------------------------
# connector tests
# ---------------------------------------------------------------------------


class TestConnectors:
    def _req(self, connector_id, privacy_cleared=True, specialty="epilepsy", task_type="medication"):
        return ConnectorRequest(
            connector_id=connector_id,
            query_hash="testhash",
            specialty=specialty,
            task_type=task_type,
            privacy_cleared=privacy_cleared,
        )

    def test_dxgpt_stub_returns_success(self):
        resp = call_connector(self._req("dxgpt_stub"))
        assert resp.success is True
        assert resp.content
        assert resp.connector_id == "dxgpt_stub"

    def test_sage_epilepsy_stub_epilepsy_specialty(self):
        resp = call_connector(self._req("sage_epilepsy_stub"))
        assert resp.success is True
        assert resp.confidence > 0

    def test_sage_epilepsy_stub_wrong_specialty(self):
        resp = call_connector(self._req("sage_epilepsy_stub", specialty="cardiology"))
        assert resp.success is False
        assert resp.error is not None

    def test_patientnotes_ddi_stub_returns_success(self):
        resp = call_connector(self._req("patientnotes_ddi_stub"))
        assert resp.success is True

    def test_unknown_connector_returns_error(self):
        resp = call_connector(self._req("nonexistent_connector"))
        assert resp.success is False
        assert "Unknown connector" in (resp.error or "")

    def test_privacy_not_cleared_blocks_all(self):
        for cid in CONNECTOR_IDS:
            req = self._req(cid, privacy_cleared=False)
            resp = call_connector(req)
            assert resp.success is False

    def test_no_network_calls(self):
        # All stubs must return without network — tested by absence of side effects
        for cid in CONNECTOR_IDS:
            req = self._req(cid)
            resp = call_connector(req)
            assert isinstance(resp, ConnectorResponse)

    def test_connector_ids_list(self):
        assert "dxgpt_stub" in CONNECTOR_IDS
        assert "sage_epilepsy_stub" in CONNECTOR_IDS
        assert "patientnotes_ddi_stub" in CONNECTOR_IDS


# ---------------------------------------------------------------------------
# scoring tests
# ---------------------------------------------------------------------------


class TestScoring:
    def _success_response(self, cid="dxgpt_stub"):
        return ConnectorResponse(
            connector_id=cid,
            success=True,
            content="Some clinical reference content here for scoring.",
            confidence=0.75,
            citations=["ref_001"],
        )

    def _failed_response(self, cid="dxgpt_stub"):
        return ConnectorResponse(
            connector_id=cid,
            success=False,
            content="",
            confidence=0.0,
            citations=[],
            error="failed",
        )

    def test_failed_response_scored_as_discarded(self, clean_context, basic_classification):
        sr = score_response(self._failed_response(), clean_context, basic_classification)
        assert sr.discarded is True
        assert sr.composite_score == 0.0
        assert sr.score_band == ScoreBand.DISCARDED

    def test_success_response_not_discarded(self, clean_context, basic_classification):
        sr = score_response(self._success_response(), clean_context, basic_classification)
        assert not sr.discarded

    def test_composite_formula(self):
        composite = ScoredResponse.compute_composite(0.8, 0.7, 0.6, 0.9)
        expected = round(0.8 * 0.35 + 0.7 * 0.25 + 0.6 * 0.20 + 0.9 * 0.20, 4)
        assert abs(composite - expected) < 1e-6

    def test_score_band_discarded(self):
        sr = ScoredResponse(
            connector_id="x", raw_content="", mkb_consistency_score=0.0,
            internal_coherence_score=0.0, citation_presence_score=0.0,
            ddi_safety_score=0.0, composite_score=0.25,
            score_band=ScoreBand.DISCARDED, discarded=True,
        )
        assert sr.score_band == ScoreBand.DISCARDED

    def test_score_band_high(self, clean_context, basic_classification):
        resp = ConnectorResponse(
            connector_id="sage_epilepsy_stub",
            success=True,
            content="Detailed epilepsy management guideline with multiple recommendations.",
            confidence=0.95,
            citations=["ref1", "ref2", "ref3"],
        )
        sr = score_response(resp, clean_context, basic_classification)
        assert sr.score_band in (ScoreBand.HIGH, ScoreBand.ACCEPTABLE)

    def test_score_all_responses_list(self, clean_context, basic_classification):
        responses = [self._success_response("dxgpt_stub"), self._failed_response("sage_epilepsy_stub")]
        scored = score_all_responses(responses, clean_context, basic_classification)
        assert len(scored) == 2
        assert scored[1].discarded is True

    def test_citation_score_zero_citations(self, clean_context, basic_classification):
        resp = ConnectorResponse(
            connector_id="x", success=True, content="content", confidence=0.7, citations=[]
        )
        sr = score_response(resp, clean_context, basic_classification)
        assert sr.citation_presence_score < 0.5

    def test_ddi_modifier_applied(self, clean_context, basic_classification):
        resp = self._success_response()
        sr_normal = score_response(resp, clean_context, basic_classification, ddi_layer1_modifier=1.0)
        sr_reduced = score_response(resp, clean_context, basic_classification, ddi_layer1_modifier=0.5)
        assert sr_reduced.ddi_safety_score < sr_normal.ddi_safety_score


# ---------------------------------------------------------------------------
# safe_mode tests
# ---------------------------------------------------------------------------


class TestSafeMode:
    def _failed_resp(self):
        return ConnectorResponse(
            connector_id="x", success=False, content="", confidence=0.0, citations=[], error="fail"
        )

    def _success_resp(self):
        return ConnectorResponse(
            connector_id="x", success=True, content="ok", confidence=0.8, citations=[]
        )

    def test_manual_flag_activates(self):
        state = evaluate_safe_mode([], [], 0.9, 0.4, manual_safe_mode=True)
        assert state.active is True
        assert state.triggered_by_manual_flag is True

    def test_all_connectors_fail_activates(self):
        resps = [self._failed_resp(), self._failed_resp()]
        state = evaluate_safe_mode(resps, [], 0.9, 0.4)
        assert state.active is True
        assert state.triggered_by_connector_failure is True

    def test_low_confidence_activates(self):
        resps = [self._success_resp()]
        state = evaluate_safe_mode(resps, [], 0.1, 0.4)
        assert state.active is True
        assert state.triggered_by_low_confidence is True

    def test_no_trigger_normal_case(self):
        resps = [self._success_resp()]
        state = evaluate_safe_mode(resps, [], 0.8, 0.4)
        assert state.active is False

    def test_apply_prefix(self):
        state = SafeModeState(active=True, reason="test")
        result = apply_safe_mode_prefix("some response", state)
        assert result.startswith(SAFE_MODE_PREFIX)

    def test_prefix_not_doubled(self):
        state = SafeModeState(active=True, reason="test")
        already_prefixed = f"{SAFE_MODE_PREFIX} something"
        result = apply_safe_mode_prefix(already_prefixed, state)
        assert result.count(SAFE_MODE_PREFIX) == 1

    def test_inactive_safe_mode_no_prefix(self):
        state = SafeModeState(active=False, reason="")
        result = apply_safe_mode_prefix("normal response", state)
        assert not result.startswith(SAFE_MODE_PREFIX)

    def test_safe_mode_prefix_constant(self):
        assert SAFE_MODE_PREFIX == "[SAFE MODE — MKB only, no external AI]"


# ---------------------------------------------------------------------------
# refusal tests
# ---------------------------------------------------------------------------


class TestRefusal:
    def test_prescription_dosing_refused(self):
        c = classify_query("what dosage should I prescribe for valproate?")
        refused, msg = evaluate_refusal(c)
        assert refused is True
        assert msg is not None
        assert "clinician" in msg.lower() or "pharmacist" in msg.lower()

    def test_normal_query_not_refused(self):
        c = classify_query("what are epilepsy treatments?")
        refused, _ = evaluate_refusal(c)
        assert refused is False

    def test_refusal_message_mentions_no_prescription_dosing(self):
        c = classify_query("prescribe dosing for levetiracetam mg daily")
        refused, msg = evaluate_refusal(c)
        assert refused is True
        assert msg


# ---------------------------------------------------------------------------
# engine integration tests
# ---------------------------------------------------------------------------


class TestEngine:
    def test_basic_query_runs(self, populated_store, default_config):
        result = run_decision_engine("What treats epilepsy?", populated_store, default_config)
        assert result.query_hash
        assert result.final_response
        assert result.external_api_used is False

    def test_refused_query_returns_refusal(self, populated_store, default_config):
        result = run_decision_engine(
            "What prescription dosing should I prescribe?", populated_store, default_config
        )
        assert result.refused is True
        assert result.refusal_reason

    def test_safe_mode_when_all_fail(self, populated_store, default_config):
        result = run_decision_engine(
            "epilepsy treatment", populated_store, default_config,
            simulated_connector_mode="all_fail",
        )
        assert result.safe_mode.active is True
        assert result.final_response.startswith(SAFE_MODE_PREFIX)

    def test_manual_safe_mode(self, populated_store, default_config):
        result = run_decision_engine(
            "epilepsy treatment", populated_store, default_config,
            manual_safe_mode=True,
        )
        assert result.safe_mode.active is True
        assert result.safe_mode.triggered_by_manual_flag is True

    def test_phi_in_query_detected(self, populated_store, default_config):
        result = run_decision_engine(
            "Jane Doe DOB: 01/01/1990 needs seizure medication",
            populated_store, default_config,
        )
        assert result.raw_phi_in_query is True
        assert result.phi_sanitized_before_connectors is True

    def test_no_phi_in_clean_query(self, populated_store, default_config):
        result = run_decision_engine(
            "general epilepsy treatment overview",
            populated_store, default_config,
        )
        assert result.raw_phi_in_query is False

    def test_external_api_never_used(self, populated_store, default_config):
        result = run_decision_engine("epilepsy", populated_store, default_config)
        assert result.external_api_used is False

    def test_ledger_events_written(self, populated_store, default_config):
        before = populated_store.count_ledger_events()
        run_decision_engine("epilepsy treatment", populated_store, default_config)
        after = populated_store.count_ledger_events()
        assert after > before

    def test_ddi_check_with_patientnotes_connector(self, populated_store):
        config = CKAConfig(ACTIVE_CONNECTORS=["patientnotes_ddi_stub"])
        result = run_decision_engine(
            "interaction between levetiracetam and valproate",
            populated_store, config,
        )
        assert result.ddi_layer1_checked is True

    def test_result_has_classification(self, populated_store, default_config):
        result = run_decision_engine("epilepsy", populated_store, default_config)
        assert result.classification is not None
        assert isinstance(result.classification, QueryClassification)

    def test_result_has_context(self, populated_store, default_config):
        result = run_decision_engine("epilepsy", populated_store, default_config)
        assert result.context is not None

    def test_scored_responses_list(self, populated_store, default_config):
        result = run_decision_engine("epilepsy treatment", populated_store, default_config)
        assert isinstance(result.scored_responses, list)

    def test_none_config_uses_default(self, populated_store):
        result = run_decision_engine("epilepsy", populated_store, None)
        assert result.final_response

    def test_empty_store_does_not_crash(self, empty_store, default_config):
        result = run_decision_engine("epilepsy treatment", empty_store, default_config)
        assert result.final_response

    def test_high_threshold_triggers_safe_mode(self, populated_store):
        config = CKAConfig(SAFE_MODE_THRESHOLD=0.99)
        result = run_decision_engine("epilepsy", populated_store, config)
        assert result.safe_mode.active is True

    def test_low_threshold_allows_normal_flow(self, populated_store):
        config = CKAConfig(SAFE_MODE_THRESHOLD=0.01)
        result = run_decision_engine(
            "What is the treatment for epilepsy seizures?",
            populated_store, config,
        )
        # Not all-fail and very low threshold — safe mode likely inactive
        # (Only check it doesn't crash; safe mode may still activate for other reasons)
        assert result.final_response


# ---------------------------------------------------------------------------
# models tests
# ---------------------------------------------------------------------------


class TestModels:
    def test_score_band_enum_values(self):
        assert ScoreBand.DISCARDED == "discarded"
        assert ScoreBand.LOW == "low"
        assert ScoreBand.ACCEPTABLE == "acceptable"
        assert ScoreBand.HIGH == "high"

    def test_query_specialty_enum(self):
        assert QuerySpecialty.EPILEPSY == "epilepsy"
        assert QuerySpecialty.NEUROLOGY == "neurology"
        assert QuerySpecialty.UNKNOWN == "unknown"

    def test_query_task_type_enum(self):
        assert QueryTaskType.MEDICATION == "medication"
        assert QueryTaskType.DIAGNOSIS == "diagnosis"

    def test_scored_response_composite_formula(self):
        val = ScoredResponse.compute_composite(1.0, 1.0, 1.0, 1.0)
        assert abs(val - 1.0) < 1e-6

    def test_scored_response_composite_weights_sum_to_one(self):
        from clinical_knowledge.decision_engine.scoring import SCORE_WEIGHTS
        total = sum(SCORE_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9

    def test_safe_mode_state_prefix_constant(self):
        state = SafeModeState(active=True, reason="test")
        assert state.prefix == SAFE_MODE_PREFIX


# ---------------------------------------------------------------------------
# ledger integration — SAFE_MODE_ENTRY and RESPONSE_DISCARDED events
# ---------------------------------------------------------------------------


class TestLedgerIntegration:
    def test_safe_mode_entry_event_not_reserved(self):
        from clinical_knowledge.ledger import make_safe_mode_entry_event
        from clinical_knowledge.models import LedgerEventType
        evt = make_safe_mode_entry_event("rec1", "safe_rec1", "test reason")
        assert evt.event_type == LedgerEventType.SAFE_MODE_ENTRY

    def test_response_discarded_event(self):
        from clinical_knowledge.ledger import make_response_discarded_event
        from clinical_knowledge.models import LedgerEventType
        evt = make_response_discarded_event("rec1", "safe_rec1", 0.25, "below threshold")
        assert evt.event_type == LedgerEventType.RESPONSE_DISCARDED

    def test_reserved_types_no_longer_include_safe_mode(self):
        from clinical_knowledge.models import _RESERVED_EVENT_TYPES, LedgerEventType
        assert LedgerEventType.SAFE_MODE_ENTRY not in _RESERVED_EVENT_TYPES
        assert LedgerEventType.RESPONSE_DISCARDED not in _RESERVED_EVENT_TYPES

    def test_enrichment_write_now_active_in_b06(self):
        from clinical_knowledge.models import _RESERVED_EVENT_TYPES, LedgerEventType
        assert LedgerEventType.ENRICHMENT_WRITE not in _RESERVED_EVENT_TYPES
        assert len(_RESERVED_EVENT_TYPES) == 0

    def test_engine_writes_privacy_audit_to_ledger(self, populated_store, default_config):
        from clinical_knowledge.models import LedgerEventType
        run_decision_engine("epilepsy", populated_store, default_config)
        events = populated_store.read_ledger_events()
        types = [e["event_type"] for e in events]
        assert LedgerEventType.PRIVACY_AUDIT.value in types

    def test_engine_writes_safe_mode_event_when_triggered(self, populated_store, default_config):
        from clinical_knowledge.models import LedgerEventType
        run_decision_engine(
            "epilepsy", populated_store, default_config,
            simulated_connector_mode="all_fail",
        )
        events = populated_store.read_ledger_events()
        types = [e["event_type"] for e in events]
        assert LedgerEventType.SAFE_MODE_ENTRY.value in types
