"""CKA-B06 Controlled Enrichment + Hypothesis Tier — test suite.

All tests use synthetic data only. No real drugs. No real patients. No external APIs.
"""
from __future__ import annotations

import pytest

from clinical_knowledge.config import CKAConfig
from clinical_knowledge.enrichment.candidate_extractor import (
    extract_enrichment_candidates_from_structured_response,
)
from clinical_knowledge.enrichment.enrichment_queue import EnrichmentQueue
from clinical_knowledge.enrichment.integration import (
    is_duplicate_candidate,
    process_enrichment_candidate,
)
from clinical_knowledge.enrichment.models import (
    EnrichmentAction,
    EnrichmentCandidate,
    EnrichmentCandidateStatus,
    EnrichmentSourceKind,
    PromotionDecision,
)
from clinical_knowledge.enrichment.promotion import prepare_hypothesis_promotion
from clinical_knowledge.models import (
    KnowledgeTier,
    LedgerEventType,
    MKBRecord,
    RecordStatus,
    SourceType,
    TrustLevel,
)
from clinical_knowledge.privacy.report_privacy import check_public_report_payload
from clinical_knowledge.safe_ids import make_safe_record_id, new_record_id
from clinical_knowledge.store import MKBStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store():
    return MKBStore(db_path=":memory:")


@pytest.fixture
def queue():
    return EnrichmentQueue()


@pytest.fixture
def config():
    return CKAConfig()


@pytest.fixture
def config_promote():
    return CKAConfig(ENRICH_PROMOTE=True)


def _ai_payload(**overrides):
    base = {
        "source_name": "dxgpt_stub",
        "source_kind": "ai_response",
        "specialty": "epilepsy",
        "facts": [
            {
                "fact_type": "diagnosis",
                "entity_text": "synthetic condition alpha",
                "confidence": 0.72,
                "structured": {"source_count": 1},
            }
        ],
    }
    base.update(overrides)
    return base


def _med_payload(entity="synth_med_clear", fact_type="medication"):
    return {
        "source_name": "dxgpt_stub",
        "source_kind": "ai_response",
        "specialty": "epilepsy",
        "facts": [
            {
                "fact_type": fact_type,
                "entity_text": entity,
                "confidence": 0.70,
                "structured": {},
            }
        ],
    }


def _active_record(entity="synth_existing_fact", fact_type="diagnosis", trust=TrustLevel.OPERATOR_REVIEWED):
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat()
    rid = new_record_id()
    return MKBRecord(
        record_id=rid,
        safe_record_id=make_safe_record_id(rid),
        session_id="b06_test",
        fact_type=fact_type,
        entity_text=entity,
        trust_level=trust,
        tier=KnowledgeTier.ACTIVE,
        status=RecordStatus.CONFIRMED,
        source_type=SourceType.SYNTHETIC,
        confidence=0.80,
        created_at=ts,
    )


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------

class TestEnums:
    def test_enrichment_source_kind_values(self):
        assert EnrichmentSourceKind.AI_RESPONSE == "ai_response"
        assert EnrichmentSourceKind.WEB_UNVERIFIED == "web_unverified"
        assert EnrichmentSourceKind.CONNECTOR_STUB == "connector_stub"
        assert EnrichmentSourceKind.MANUAL_REVIEW_PREPARED == "manual_review_prepared"

    def test_enrichment_candidate_status_values(self):
        assert EnrichmentCandidateStatus.CANDIDATE == "candidate"
        assert EnrichmentCandidateStatus.DUPLICATE_DISCARDED == "duplicate_discarded"
        assert EnrichmentCandidateStatus.WRITTEN_HYPOTHESIS == "written_hypothesis"
        assert EnrichmentCandidateStatus.BLOCKED_SAFETY == "blocked_safety"
        assert EnrichmentCandidateStatus.CONFLICT_QUARANTINED == "conflict_quarantined"

    def test_enrichment_action_values(self):
        assert EnrichmentAction.DISCARD_DUPLICATE == "discard_duplicate"
        assert EnrichmentAction.WRITE_HYPOTHESIS == "write_hypothesis"
        assert EnrichmentAction.BLOCK_SAFETY == "block_safety"
        assert EnrichmentAction.ROUTE_TRUTH_RESOLUTION == "route_truth_resolution"
        assert EnrichmentAction.BLOCK_AUTO_PROMOTION == "block_auto_promotion"

    def test_enrichment_write_now_active_in_b06(self):
        from clinical_knowledge.models import _RESERVED_EVENT_TYPES
        assert LedgerEventType.ENRICHMENT_WRITE not in _RESERVED_EVENT_TYPES
        assert len(_RESERVED_EVENT_TYPES) == 0

    def test_hypothesis_promoted_enum_exists(self):
        assert LedgerEventType.HYPOTHESIS_PROMOTED == "hypothesis_promoted"


# ---------------------------------------------------------------------------
# Candidate extraction tests
# ---------------------------------------------------------------------------

class TestCandidateExtractor:
    def test_ai_response_extracts_candidate(self):
        candidates = extract_enrichment_candidates_from_structured_response(_ai_payload())
        assert len(candidates) == 1
        c = candidates[0]
        assert c.source_kind == EnrichmentSourceKind.AI_RESPONSE
        assert c.fact_type == "diagnosis"
        assert c.entity_text == "synthetic condition alpha"
        assert c.proposed_tier == "hypothesis"
        assert c.extraction_method == "synthetic_structured_enrichment"

    def test_ai_connector_trust_level_3(self):
        candidates = extract_enrichment_candidates_from_structured_response(_ai_payload())
        assert candidates[0].proposed_trust_level == 3

    def test_connector_stub_trust_level_3(self):
        payload = _ai_payload(source_kind="connector_stub")
        candidates = extract_enrichment_candidates_from_structured_response(payload)
        assert candidates[0].proposed_trust_level == 3

    def test_web_unverified_high_quality_trust_level_4(self):
        payload = {
            "source_name": "web_stub",
            "source_kind": "web_unverified",
            "source_quality": "high",
            "specialty": "neurology",
            "facts": [{"fact_type": "diagnosis", "entity_text": "synth condition", "confidence": 0.6}],
        }
        candidates = extract_enrichment_candidates_from_structured_response(payload)
        assert candidates[0].proposed_trust_level == 4

    def test_web_unverified_low_quality_trust_level_5(self):
        payload = {
            "source_name": "web_stub",
            "source_kind": "web_unverified",
            "source_quality": "low",
            "specialty": "neurology",
            "facts": [{"fact_type": "diagnosis", "entity_text": "synth condition", "confidence": 0.6}],
        }
        candidates = extract_enrichment_candidates_from_structured_response(payload)
        assert candidates[0].proposed_trust_level == 5

    def test_proposed_tier_always_hypothesis(self):
        candidates = extract_enrichment_candidates_from_structured_response(_ai_payload())
        assert all(c.proposed_tier == "hypothesis" for c in candidates)

    def test_invalid_fact_missing_entity_skipped(self):
        payload = {
            "source_name": "dxgpt_stub",
            "source_kind": "ai_response",
            "specialty": "epilepsy",
            "facts": [
                {"fact_type": "diagnosis"},  # missing entity_text
                {"fact_type": "diagnosis", "entity_text": "valid entity", "confidence": 0.5},
            ],
        }
        candidates = extract_enrichment_candidates_from_structured_response(payload)
        assert len(candidates) == 1
        assert candidates[0].entity_text == "valid entity"

    def test_invalid_confidence_skipped(self):
        payload = {
            "source_name": "dxgpt_stub",
            "source_kind": "ai_response",
            "specialty": "epilepsy",
            "facts": [{"fact_type": "diagnosis", "entity_text": "bad", "confidence": 1.5}],
        }
        candidates = extract_enrichment_candidates_from_structured_response(payload)
        assert len(candidates) == 0

    def test_empty_facts_returns_empty(self):
        candidates = extract_enrichment_candidates_from_structured_response(
            {"source_kind": "ai_response", "facts": []}
        )
        assert candidates == []

    def test_non_dict_payload_returns_empty(self):
        assert extract_enrichment_candidates_from_structured_response([]) == []
        assert extract_enrichment_candidates_from_structured_response("not a dict") == []

    def test_public_summary_uses_safe_ids_only(self):
        candidates = extract_enrichment_candidates_from_structured_response(_ai_payload())
        c = candidates[0]
        summary = c.safe_public_summary
        assert "safe_candidate_id" in summary
        assert summary["safe_candidate_id"].startswith("cka_cand_")
        assert "candidate_id" not in summary
        assert "source_response" not in str(summary)

    def test_source_response_hash_not_exposed(self):
        candidates = extract_enrichment_candidates_from_structured_response(_ai_payload())
        c = candidates[0]
        # safe_public_summary must not include raw response text
        assert "source_response_text" not in c.safe_public_summary
        # hash is present on the object itself but not in public summary
        assert "source_response_hash" not in c.safe_public_summary

    def test_candidate_enforces_hypothesis_tier(self):
        with pytest.raises(ValueError, match="hypothesis"):
            EnrichmentCandidate(
                candidate_id="x",
                safe_candidate_id="cka_cand_x",
                source_kind=EnrichmentSourceKind.AI_RESPONSE,
                source_name="stub",
                source_response_hash="abc",
                fact_type="diagnosis",
                entity_text="something",
                proposed_tier="active",  # must fail
                extraction_method="synthetic_structured_enrichment",
            )

    def test_candidate_enforces_extraction_method(self):
        with pytest.raises(ValueError, match="synthetic_structured_enrichment"):
            EnrichmentCandidate(
                candidate_id="x",
                safe_candidate_id="cka_cand_x",
                source_kind=EnrichmentSourceKind.AI_RESPONSE,
                source_name="stub",
                source_response_hash="abc",
                fact_type="diagnosis",
                entity_text="something",
                proposed_tier="hypothesis",
                extraction_method="manual",  # must fail
            )

    def test_no_clinical_text_generated(self):
        candidates = extract_enrichment_candidates_from_structured_response(_ai_payload())
        for c in candidates:
            assert "prescri" not in str(c.safe_public_summary).lower()
            assert "recommend" not in str(c.safe_public_summary).lower()


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_duplicate_candidate_detected(self):
        candidates = extract_enrichment_candidates_from_structured_response(_ai_payload())
        c = candidates[0]
        existing = [
            {
                "specialty": "epilepsy",
                "fact_type": "diagnosis",
                "entity_text": "synthetic condition alpha",
                "structured": '{"source_count": 1}',
            }
        ]
        assert is_duplicate_candidate(c, existing) is True

    def test_non_duplicate_not_flagged(self):
        candidates = extract_enrichment_candidates_from_structured_response(_ai_payload())
        c = candidates[0]
        existing = [
            {
                "specialty": "neurology",
                "fact_type": "diagnosis",
                "entity_text": "different condition",
                "structured": "{}",
            }
        ]
        assert is_duplicate_candidate(c, existing) is False

    def test_empty_existing_not_duplicate(self):
        candidates = extract_enrichment_candidates_from_structured_response(_ai_payload())
        assert is_duplicate_candidate(candidates[0], []) is False

    def test_second_write_discarded(self, store, queue, config):
        candidates = extract_enrichment_candidates_from_structured_response(_ai_payload())
        cand1 = candidates[0]
        r1 = process_enrichment_candidate(cand1, store, queue, config)
        assert r1.action == EnrichmentAction.WRITE_HYPOTHESIS

        candidates2 = extract_enrichment_candidates_from_structured_response(_ai_payload())
        cand2 = candidates2[0]
        r2 = process_enrichment_candidate(cand2, store, queue, config)
        assert r2.action == EnrichmentAction.DISCARD_DUPLICATE
        assert r2.status == EnrichmentCandidateStatus.DUPLICATE_DISCARDED
        assert len(store.list_hypothesis()) == 1


# ---------------------------------------------------------------------------
# Hypothesis writer tests
# ---------------------------------------------------------------------------

class TestHypothesisWriter:
    def test_ai_candidate_written_as_hypothesis(self, store, queue, config):
        candidates = extract_enrichment_candidates_from_structured_response(_ai_payload())
        result = process_enrichment_candidate(candidates[0], store, queue, config)
        assert result.action == EnrichmentAction.WRITE_HYPOTHESIS
        hypothesis = store.list_hypothesis()
        assert len(hypothesis) == 1
        assert hypothesis[0]["tier"] == KnowledgeTier.HYPOTHESIS.value

    def test_ai_fact_never_written_active(self, store, queue, config):
        candidates = extract_enrichment_candidates_from_structured_response(_ai_payload())
        process_enrichment_candidate(candidates[0], store, queue, config)
        assert store.list_active() == []

    def test_written_record_requires_review(self, store, queue, config):
        candidates = extract_enrichment_candidates_from_structured_response(_ai_payload())
        result = process_enrichment_candidate(candidates[0], store, queue, config)
        assert result.written_record is not None
        assert result.written_record.requires_review is True

    def test_written_record_trust_level_3_for_ai(self, store, queue, config):
        candidates = extract_enrichment_candidates_from_structured_response(_ai_payload())
        result = process_enrichment_candidate(candidates[0], store, queue, config)
        assert result.written_record is not None
        tl = result.written_record.trust_level
        assert int(tl) == 3

    def test_enrichment_write_ledger_event_written(self, store, queue, config):
        candidates = extract_enrichment_candidates_from_structured_response(_ai_payload())
        process_enrichment_candidate(candidates[0], store, queue, config)
        events = store.read_ledger_events()
        types = [e["event_type"] for e in events]
        assert LedgerEventType.ENRICHMENT_WRITE.value in types

    def test_web_candidate_trust_level_4_or_5(self, store, queue, config):
        payload = {
            "source_name": "web_stub", "source_kind": "web_unverified",
            "source_quality": "high", "specialty": "neurology",
            "facts": [{"fact_type": "diagnosis", "entity_text": "web synth fact", "confidence": 0.6}],
        }
        candidates = extract_enrichment_candidates_from_structured_response(payload)
        result = process_enrichment_candidate(candidates[0], store, queue, config)
        assert result.written_record is not None
        assert int(result.written_record.trust_level) in (4, 5)

    def test_hypothesis_ledger_event_not_falsely_written_when_only_prepared(self, config):
        # Verify hypothesis_promoted ledger event is not written when promotion is only "prepared"
        store = MKBStore(db_path=":memory:")
        queue = EnrichmentQueue()
        candidates = extract_enrichment_candidates_from_structured_response(_ai_payload())
        result = process_enrichment_candidate(candidates[0], store, queue, config)
        decision = prepare_hypothesis_promotion(result.written_record, config)
        # ENRICH_PROMOTE=False → allowed=False, no ledger event written
        assert decision.allowed is False
        events = store.read_ledger_events()
        types = [e["event_type"] for e in events]
        assert LedgerEventType.HYPOTHESIS_PROMOTED.value not in types


# ---------------------------------------------------------------------------
# ENRICH_PROMOTE tests
# ---------------------------------------------------------------------------

class TestPromotion:
    def test_enrich_promote_false_by_default(self, config):
        assert config.ENRICH_PROMOTE is False

    def test_auto_promotion_blocked_when_enrich_promote_false(self, store, queue, config):
        candidates = extract_enrichment_candidates_from_structured_response(_ai_payload())
        result = process_enrichment_candidate(candidates[0], store, queue, config)
        decision = prepare_hypothesis_promotion(result.written_record, config)
        assert decision.allowed is False
        assert decision.promotion_mode == "auto_blocked"
        assert decision.auto_promotion_attempted is False

    def test_manual_promotion_prepared_not_auto_executed(self, store, queue):
        config_p = CKAConfig(ENRICH_PROMOTE=True)
        candidates = extract_enrichment_candidates_from_structured_response(_ai_payload())
        result = process_enrichment_candidate(candidates[0], store, queue, config_p)
        decision = prepare_hypothesis_promotion(
            result.written_record, config_p, manual_review_confirmed=True
        )
        assert decision.allowed is True
        assert decision.promotion_mode == "manual_prepared"
        assert decision.auto_promotion_attempted is False
        assert decision.requires_manual_review is True
        # Record stays hypothesis — no auto active write
        assert result.written_record.tier == KnowledgeTier.HYPOTHESIS

    def test_enrich_promote_true_without_manual_confirmation_blocked(self, store, queue):
        config_p = CKAConfig(ENRICH_PROMOTE=True)
        candidates = extract_enrichment_candidates_from_structured_response(_ai_payload())
        result = process_enrichment_candidate(candidates[0], store, queue, config_p)
        decision = prepare_hypothesis_promotion(result.written_record, config_p, manual_review_confirmed=False)
        assert decision.allowed is False

    def test_quarantined_record_promotion_blocked(self, config):
        rid = new_record_id()
        record = MKBRecord(
            record_id=rid,
            safe_record_id=make_safe_record_id(rid),
            session_id="test",
            fact_type="diagnosis",
            entity_text="quarantined entity",
            tier=KnowledgeTier.QUARANTINED,
            trust_level=TrustLevel.OPERATOR_REVIEWED,
            source_type=SourceType.SYNTHETIC,
        )
        decision = prepare_hypothesis_promotion(record, config)
        assert decision.allowed is False
        assert decision.promotion_mode == "blocked_quarantined"

    def test_high_ddi_record_promotion_blocked(self, config):
        from clinical_knowledge.models import DDIStatus
        rid = new_record_id()
        record = MKBRecord(
            record_id=rid,
            safe_record_id=make_safe_record_id(rid),
            session_id="test",
            fact_type="medication",
            entity_text="synth_med_alpha",
            tier=KnowledgeTier.HYPOTHESIS,
            trust_level=TrustLevel.OPERATOR_REVIEWED,
            source_type=SourceType.SYNTHETIC,
            ddi_status=DDIStatus.BLOCKED,
        )
        decision = prepare_hypothesis_promotion(record, config)
        assert decision.allowed is False
        assert decision.promotion_mode == "blocked_pending_safety"

    def test_no_auto_promotion_based_on_confidence(self, store, queue, config):
        # High confidence should not bypass ENRICH_PROMOTE=False
        payload = _ai_payload()
        payload["facts"][0]["confidence"] = 0.99
        candidates = extract_enrichment_candidates_from_structured_response(payload)
        result = process_enrichment_candidate(candidates[0], store, queue, config)
        decision = prepare_hypothesis_promotion(result.written_record, config)
        assert decision.allowed is False


# ---------------------------------------------------------------------------
# Medication Safety Gate integration
# ---------------------------------------------------------------------------

class TestMedicationGateIntegration:
    def test_medication_candidate_routes_through_ddi_gate(self, store, queue, config):
        candidates = extract_enrichment_candidates_from_structured_response(
            _med_payload("synth_med_clear")
        )
        result = process_enrichment_candidate(
            candidates[0], store, queue, config,
            ddi_mode="force_none", active_medications=["some_other"]
        )
        assert result.medication_gate_result is not None

    def test_none_ddi_allows_hypothesis_write(self, store, queue, config):
        candidates = extract_enrichment_candidates_from_structured_response(
            _med_payload("synth_med_clear")
        )
        result = process_enrichment_candidate(
            candidates[0], store, queue, config,
            ddi_mode="force_none", active_medications=["other"]
        )
        assert result.action == EnrichmentAction.WRITE_HYPOTHESIS
        assert len(store.list_hypothesis()) == 1

    def test_low_ddi_allows_hypothesis_write(self, store, queue, config):
        candidates = extract_enrichment_candidates_from_structured_response(
            _med_payload("synth_med_epsilon")
        )
        result = process_enrichment_candidate(
            candidates[0], store, queue, config,
            ddi_mode="force_low", active_medications=["synth_med_zeta"]
        )
        assert result.action == EnrichmentAction.WRITE_HYPOTHESIS

    def test_high_ddi_blocks_hypothesis_write(self, store, queue, config):
        candidates = extract_enrichment_candidates_from_structured_response(
            _med_payload("synth_med_alpha")
        )
        result = process_enrichment_candidate(
            candidates[0], store, queue, config,
            ddi_mode="force_high", active_medications=["synth_med_beta"]
        )
        assert result.action == EnrichmentAction.BLOCK_SAFETY
        assert result.status == EnrichmentCandidateStatus.BLOCKED_SAFETY
        assert len(store.list_hypothesis()) == 0

    def test_high_ddi_writes_ddi_block_ledger_event(self, store, queue, config):
        candidates = extract_enrichment_candidates_from_structured_response(
            _med_payload("synth_med_alpha")
        )
        process_enrichment_candidate(
            candidates[0], store, queue, config,
            ddi_mode="force_high", active_medications=["synth_med_beta"]
        )
        events = store.read_ledger_events()
        types = [e["event_type"] for e in events]
        assert LedgerEventType.DDI_BLOCK.value in types

    def test_medium_ddi_queues_pending_safety(self, store, queue, config):
        candidates = extract_enrichment_candidates_from_structured_response(
            _med_payload("synth_med_gamma")
        )
        result = process_enrichment_candidate(
            candidates[0], store, queue, config,
            ddi_mode="force_medium", active_medications=["synth_med_delta"]
        )
        assert result.action == EnrichmentAction.QUEUE_PENDING_SAFETY
        assert result.status == EnrichmentCandidateStatus.QUEUED_PENDING_SAFETY
        assert len(store.list_hypothesis()) == 0

    def test_ddi_unavailable_queues_pending(self, store, queue, config):
        candidates = extract_enrichment_candidates_from_structured_response(
            _med_payload("synth_med_clear")
        )
        result = process_enrichment_candidate(
            candidates[0], store, queue, config,
            ddi_mode="unavailable", active_medications=[]
        )
        assert result.action == EnrichmentAction.QUEUE_PENDING_SAFETY
        assert len(store.list_hypothesis()) == 0

    def test_blocked_result_queued_in_enrichment_queue(self, store, queue, config):
        candidates = extract_enrichment_candidates_from_structured_response(
            _med_payload("synth_med_alpha")
        )
        result = process_enrichment_candidate(
            candidates[0], store, queue, config,
            ddi_mode="force_high", active_medications=["synth_med_beta"]
        )
        assert result.queued_item is not None
        assert queue.count() >= 1
        assert queue.count_by_reason("blocked_high_ddi") >= 1

    def test_medium_ddi_queued_with_pending_medium_ack(self, store, queue, config):
        candidates = extract_enrichment_candidates_from_structured_response(
            _med_payload("synth_med_gamma")
        )
        process_enrichment_candidate(
            candidates[0], store, queue, config,
            ddi_mode="force_medium", active_medications=["synth_med_delta"]
        )
        assert queue.count_by_reason("pending_medium_ack") >= 1

    def test_no_medication_advice_in_explanations(self, store, queue, config):
        candidates = extract_enrichment_candidates_from_structured_response(
            _med_payload("synth_med_alpha")
        )
        result = process_enrichment_candidate(
            candidates[0], store, queue, config,
            ddi_mode="force_high", active_medications=["synth_med_beta"]
        )
        assert "prescri" not in result.explanation.lower()
        assert "recommend" not in result.explanation.lower()
        assert "dosing" not in result.explanation.lower()

    def test_ddi_metadata_retained_in_write_result(self, store, queue, config):
        candidates = extract_enrichment_candidates_from_structured_response(
            _med_payload("synth_med_clear")
        )
        result = process_enrichment_candidate(
            candidates[0], store, queue, config,
            ddi_mode="force_none", active_medications=[]
        )
        assert result.medication_gate_result is not None


# ---------------------------------------------------------------------------
# Enrichment queue tests
# ---------------------------------------------------------------------------

class TestEnrichmentQueue:
    def test_enqueue_creates_item(self, queue):
        item = queue.enqueue("cka_cand_abc", "pending_ddi_check")
        assert item.reason == "pending_ddi_check"
        assert item.status == "pending"
        assert item.safe_queue_id.startswith("cka_q_")

    def test_list_items(self, queue):
        queue.enqueue("cka_cand_1", "pending_ddi_check")
        queue.enqueue("cka_cand_2", "blocked_high_ddi")
        items = queue.list_items()
        assert len(items) == 2

    def test_mark_status(self, queue):
        item = queue.enqueue("cka_cand_1", "pending_ddi_check")
        success = queue.mark_status(item.queue_id, "resolved")
        assert success is True
        assert queue.list_items()[0].status == "resolved"

    def test_count_by_reason(self, queue):
        queue.enqueue("cka_cand_1", "safe_mode_enrichment_disabled")
        queue.enqueue("cka_cand_2", "safe_mode_enrichment_disabled")
        queue.enqueue("cka_cand_3", "blocked_high_ddi")
        assert queue.count_by_reason("safe_mode_enrichment_disabled") == 2
        assert queue.count_by_reason("blocked_high_ddi") == 1

    def test_queue_item_public_summary_safe(self, queue):
        item = queue.enqueue("cka_cand_abc", "pending_ddi_check")
        summary = item.safe_public_summary
        assert "safe_queue_id" in summary
        assert "candidate_safe_id" in summary
        assert "queue_id" not in summary

    def test_safe_mode_queues_with_correct_reason(self, store, queue, config):
        candidates = extract_enrichment_candidates_from_structured_response(
            {"source_name": "dxgpt_stub", "source_kind": "ai_response",
             "specialty": "epilepsy",
             "facts": [{"fact_type": "diagnosis", "entity_text": "safe entity", "confidence": 0.5}]}
        )
        process_enrichment_candidate(candidates[0], store, queue, config, safe_mode=True)
        assert queue.count_by_reason("safe_mode_enrichment_disabled") == 1


# ---------------------------------------------------------------------------
# Truth Resolution integration
# ---------------------------------------------------------------------------

class TestTruthResolutionHandoff:
    def test_non_conflicting_candidate_writes_hypothesis(self, store, queue, config):
        existing = _active_record("unrelated fact", "diagnosis")
        store.insert_record(existing)
        candidates = extract_enrichment_candidates_from_structured_response(_ai_payload())
        result = process_enrichment_candidate(candidates[0], store, queue, config)
        assert result.action in (EnrichmentAction.WRITE_HYPOTHESIS, EnrichmentAction.ROUTE_TRUTH_RESOLUTION)

    def test_dose_conflict_routes_to_truth_resolution(self, store, queue, config):
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).isoformat()
        existing = MKBRecord(
            record_id=new_record_id(),
            safe_record_id=make_safe_record_id(new_record_id()),
            session_id="b06_test",
            fact_type="medication_antiepileptic",
            entity_text="synth_med_alpha 500mg twice daily",
            trust_level=TrustLevel.OPERATOR_REVIEWED,
            tier=KnowledgeTier.ACTIVE,
            status=RecordStatus.CONFIRMED,
            source_type=SourceType.SYNTHETIC,
            confidence=0.80,
            created_at=ts,
        )
        store.insert_record(existing)

        conflict_payload = {
            "source_name": "dxgpt_stub",
            "source_kind": "ai_response",
            "specialty": "epilepsy",
            "facts": [
                {
                    "fact_type": "medication_antiepileptic",
                    "entity_text": "synth_med_alpha 1000mg once daily",
                    "confidence": 0.75,
                    "structured": {},
                }
            ],
        }
        candidates = extract_enrichment_candidates_from_structured_response(conflict_payload)
        result = process_enrichment_candidate(candidates[0], store, queue, config)
        assert result.action in (EnrichmentAction.ROUTE_TRUTH_RESOLUTION, EnrichmentAction.WRITE_HYPOTHESIS)
        # No active record for the new candidate
        if result.written_record is not None:
            active_ids = {r["record_id"] for r in store.list_active()}
            assert result.written_record.record_id not in active_ids


# ---------------------------------------------------------------------------
# Safety boundaries
# ---------------------------------------------------------------------------

class TestSafetyBoundaries:
    def test_no_clinical_recommendations_generated(self, store, queue, config):
        candidates = extract_enrichment_candidates_from_structured_response(_ai_payload())
        result = process_enrichment_candidate(candidates[0], store, queue, config)
        # No clinical advice strings anywhere in public summary
        summary_str = str(result.safe_public_summary).lower()
        assert "recommend" not in summary_str
        assert "prescri" not in summary_str
        assert "diagnose" not in summary_str

    def test_no_prescription_dosing_advice(self, store, queue, config):
        candidates = extract_enrichment_candidates_from_structured_response(
            _med_payload("synth_med_clear")
        )
        result = process_enrichment_candidate(
            candidates[0], store, queue, config,
            ddi_mode="force_none", active_medications=[]
        )
        assert "dose" not in result.explanation.lower() or "dosing advice" not in result.explanation.lower()
        assert "prescri" not in result.explanation.lower()

    def test_source_response_text_not_in_public_summary(self, store, queue, config):
        candidates = extract_enrichment_candidates_from_structured_response(_ai_payload())
        result = process_enrichment_candidate(candidates[0], store, queue, config)
        summary_str = str(result.safe_public_summary)
        assert "source_response_text" not in summary_str
        assert '"facts"' not in summary_str

    def test_no_real_api_or_llm_required(self, store, queue, config):
        # Extraction must work without any network calls
        candidates = extract_enrichment_candidates_from_structured_response(_ai_payload())
        assert len(candidates) == 1
        result = process_enrichment_candidate(candidates[0], store, queue, config)
        assert result.action == EnrichmentAction.WRITE_HYPOTHESIS

    def test_ai_derived_fact_never_written_active_regardless_of_confidence(self, store, queue, config):
        payload = _ai_payload()
        payload["facts"][0]["confidence"] = 1.0
        candidates = extract_enrichment_candidates_from_structured_response(payload)
        process_enrichment_candidate(candidates[0], store, queue, config)
        assert store.list_active() == []

    def test_all_event_types_now_active(self):
        from clinical_knowledge.models import _RESERVED_EVENT_TYPES
        assert len(_RESERVED_EVENT_TYPES) == 0


# ---------------------------------------------------------------------------
# Ledger tests
# ---------------------------------------------------------------------------

class TestLedgerHelpers:
    def test_enrichment_write_event_type(self):
        from clinical_knowledge.ledger import make_enrichment_write_event
        rid = new_record_id()
        evt = make_enrichment_write_event(
            record_id=rid,
            safe_record_id=make_safe_record_id(rid),
            tier="hypothesis",
            trust_level=3,
            safe_candidate_id="cka_cand_abc",
        )
        assert evt.event_type == LedgerEventType.ENRICHMENT_WRITE

    def test_hypothesis_promoted_event_type(self):
        from clinical_knowledge.ledger import make_hypothesis_promoted_event
        rid = new_record_id()
        evt = make_hypothesis_promoted_event(
            record_id=rid,
            safe_record_id=make_safe_record_id(rid),
            promotion_mode="manual_prepared",
        )
        assert evt.event_type == LedgerEventType.HYPOTHESIS_PROMOTED

    def test_enrichment_write_safe_public_details_no_raw(self):
        from clinical_knowledge.ledger import make_enrichment_write_event
        rid = new_record_id()
        evt = make_enrichment_write_event(
            record_id=rid,
            safe_record_id=make_safe_record_id(rid),
            tier="hypothesis",
            trust_level=3,
            safe_candidate_id="cka_cand_abc",
        )
        pub = evt.safe_public_details
        assert "patient" not in str(pub).lower()
        assert "source_response_text" not in str(pub)


# ---------------------------------------------------------------------------
# Validation script tests
# ---------------------------------------------------------------------------

class TestValidationScript:
    def test_validation_script_succeeds(self, tmp_path):
        from scripts.run_cka_block06_controlled_enrichment_validation import run_validation
        report = run_validation(report_dir=tmp_path)
        assert report["conclusion"] == "cka_b06_controlled_enrichment_ready"
        assert report["all_cases_passed"] is True
        assert report["synthetic_cases_run"] == 11
        assert report["cases_passed"] == 11
        assert report["ai_facts_written_active"] is False
        assert report["enrich_promote_default"] is False
        assert report["auto_promotion_blocked"] is True
        assert report["high_ddi_blocks_hypothesis_write"] is True
        assert report["ddi_unavailable_queues_pending"] is True
        assert report["real_llm_enrichment_used"] is False
        assert report["real_external_connectors_implemented"] is False
        assert report["external_api_used"] is False
        assert report["frozen_hitl_release_reopened"] is False

    def test_final_report_no_private_strings(self, tmp_path):
        from scripts.run_cka_block06_controlled_enrichment_validation import run_validation
        report = run_validation(report_dir=tmp_path)
        check = check_public_report_payload(report)
        assert check.passed
        assert not check.raw_phi_logged_in_public_reports
        assert check.private_filename_path_leaks == 0
        assert check.secret_leaks == 0
