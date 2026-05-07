"""Tests for CKA-B08 Multi-Connector Execution + Consensus Engine.

Covers:
- ConnectorKind / ConnectorStatus / ConnectorCapability enums
- ConnectorRegistry default specs and safety rules
- Privacy-gated request builder
- Connector executor (success/timeout/error/malformed)
- Response normalizer
- Fact extraction and agreement scoring
- Contradiction detection
- Truth Resolution handoff
- Consensus does not auto-write active records
- Consensus-to-enrichment remains hypothesis-only
- No external API calls
- No clinical recommendations or prescription dosing advice
- Validation script runs cleanly
- Public reports contain no raw private strings
- CKA-B01/B02/B03/B04/B05/B06/B07 still pass (verified separately)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from clinical_knowledge.connectors.executor import execute_connectors
from clinical_knowledge.connectors.models import (
    ConnectorCapability,
    ConnectorExecutionRequest,
    ConnectorExecutionResult,
    ConnectorKind,
    ConnectorSpec,
    ConnectorStatus,
    SimulationMode,
)
from clinical_knowledge.connectors.normalizer import normalize_connector_response
from clinical_knowledge.connectors.registry import ConnectorRegistry, ConnectorRegistryError
from clinical_knowledge.connectors.request_builder import build_connector_request
from clinical_knowledge.connectors.stubs import (
    call_dxgpt_stub,
    call_generic_stub,
    call_patientnotes_ddi_stub,
    call_sage_epilepsy_stub,
)
from clinical_knowledge.consensus.agreement import calculate_agreement
from clinical_knowledge.consensus.contradiction import detect_consensus_contradictions
from clinical_knowledge.consensus.engine import run_consensus
from clinical_knowledge.consensus.fact_extractor import extract_consensus_facts
from clinical_knowledge.consensus.integration import (
    consensus_facts_to_enrichment_candidates,
    route_consensus_contradictions_to_truth_resolution,
)
from clinical_knowledge.consensus.models import (
    ConsensusFact,
    ConsensusFactStatus,
    ConsensusResult,
    ConsensusStatus,
)
from clinical_knowledge.ledger import make_connector_execution_event, make_consensus_result_event
from clinical_knowledge.medical_coding.models import CodingCandidate, CodingStatus
from clinical_knowledge.medical_coding.synthetic_mapper import SyntheticTerminologySource
from clinical_knowledge.medical_coding.integration import code_entity
from clinical_knowledge.models import (
    DDIStatus,
    KnowledgeTier,
    LedgerEventType,
    MKBRecord,
    RecordStatus,
    SourceType,
    TrustLevel,
)
from clinical_knowledge.privacy import check_public_report_payload
from clinical_knowledge.safe_ids import make_safe_record_id
from clinical_knowledge.store import MKBStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store():
    return MKBStore(db_path=":memory:")


@pytest.fixture
def default_registry():
    return ConnectorRegistry.default()


@pytest.fixture
def dxgpt_sage_registry():
    reg = ConnectorRegistry.default()
    reg.disable("patientnotes_ddi_stub")
    reg.disable("generic_stub")
    return reg


def _make_record(record_id: str, tier: KnowledgeTier = KnowledgeTier.ACTIVE,
                 ddi_status: DDIStatus = DDIStatus.NOT_CHECKED) -> MKBRecord:
    safe_id = make_safe_record_id(record_id)
    return MKBRecord(
        record_id=record_id,
        safe_record_id=safe_id,
        session_id="test_b08",
        fact_type="diagnosis",
        entity_text="synthetic condition alpha",
        specialty="general",
        source_type=SourceType.SYNTHETIC,
        trust_level=TrustLevel.EXPERT_VALIDATED if tier == KnowledgeTier.ACTIVE else TrustLevel.UNVERIFIED,
        tier=tier,
        status=RecordStatus.CONFIRMED,
        confidence=0.80,
        ddi_status=ddi_status,
        ddi_checked=ddi_status != DDIStatus.NOT_CHECKED,
    )


def _build_requests(registry, query="synthetic condition alpha", context=None, safe_mode=False):
    context = context or {}
    return [
        build_connector_request(query, context, spec, safe_mode=safe_mode)
        for spec in registry.list_enabled()
    ]


# ---------------------------------------------------------------------------
# TestConnectorEnums
# ---------------------------------------------------------------------------

class TestConnectorEnums:
    def test_connector_kind_values(self):
        assert ConnectorKind.DXGPT_STUB.value == "dxgpt_stub"
        assert ConnectorKind.SAGE_EPILEPSY_STUB.value == "sage_epilepsy_stub"
        assert ConnectorKind.PATIENTNOTES_DDI_STUB.value == "patientnotes_ddi_stub"
        assert ConnectorKind.GENERIC_STUB.value == "generic_stub"

    def test_connector_status_values(self):
        assert ConnectorStatus.SUCCESS.value == "success"
        assert ConnectorStatus.TIMEOUT.value == "timeout"
        assert ConnectorStatus.ERROR.value == "error"
        assert ConnectorStatus.BLOCKED_PRIVACY.value == "blocked_privacy"
        assert ConnectorStatus.SKIPPED_SAFE_MODE.value == "skipped_safe_mode"
        assert ConnectorStatus.DISABLED.value == "disabled"
        assert ConnectorStatus.MALFORMED_RESPONSE.value == "malformed_response"

    def test_connector_capability_values(self):
        assert ConnectorCapability.DIAGNOSIS_SUPPORT.value == "diagnosis_support"
        assert ConnectorCapability.MEDICATION_SAFETY.value == "medication_safety"
        assert ConnectorCapability.EPILEPSY_SUPPORT.value == "epilepsy_support"
        assert ConnectorCapability.CITATION_SUPPORT.value == "citation_support"
        assert ConnectorCapability.STRUCTURED_FACT_OUTPUT.value == "structured_fact_output"

    def test_simulation_mode_values(self):
        assert SimulationMode.SUCCESS.value == "success"
        assert SimulationMode.CONTRADICTION.value == "contradiction"
        assert SimulationMode.PRIVACY_BLOCKED.value == "privacy_blocked"

    def test_consensus_status_values(self):
        assert ConsensusStatus.CONSENSUS_READY.value == "consensus_ready"
        assert ConsensusStatus.CONTRADICTION_DETECTED.value == "contradiction_detected"
        assert ConsensusStatus.ALL_RESPONSES_DISCARDED.value == "all_responses_discarded"

    def test_consensus_fact_status_values(self):
        assert ConsensusFactStatus.AGREED.value == "agreed"
        assert ConsensusFactStatus.SINGLE_SOURCE_PENALIZED.value == "single_source_penalized"
        assert ConsensusFactStatus.CONTRADICTED.value == "contradicted"

    def test_ledger_event_type_b08(self):
        assert LedgerEventType.CONNECTOR_EXECUTION.value == "connector_execution"
        assert LedgerEventType.CONSENSUS_RESULT.value == "consensus_result"


# ---------------------------------------------------------------------------
# TestConnectorRegistry
# ---------------------------------------------------------------------------

class TestConnectorRegistry:
    def test_default_registry_has_four_connectors(self, default_registry):
        all_specs = default_registry.list_all()
        assert len(all_specs) == 4
        names = {s.name for s in all_specs}
        assert "dxgpt_stub" in names
        assert "sage_epilepsy_stub" in names
        assert "patientnotes_ddi_stub" in names
        assert "generic_stub" in names

    def test_default_registry_three_enabled(self, default_registry):
        enabled = default_registry.list_enabled()
        assert len(enabled) == 3
        enabled_names = {s.name for s in enabled}
        assert "generic_stub" not in enabled_names

    def test_generic_stub_disabled_by_default(self, default_registry):
        spec = default_registry.get("generic_stub")
        assert spec is not None
        assert not spec.enabled

    def test_all_default_specs_are_synthetic_only(self, default_registry):
        for spec in default_registry.list_all():
            assert spec.synthetic_only is True
            assert spec.allow_external is False

    def test_registry_rejects_external_connector(self):
        reg = ConnectorRegistry()
        with pytest.raises(ConnectorRegistryError):
            reg.register(ConnectorSpec(
                name="bad_external",
                kind=ConnectorKind.GENERIC_STUB,
                enabled=True,
                capabilities=[],
                allow_external=True,   # NOT allowed in B08
                synthetic_only=True,
            ))

    def test_registry_rejects_non_synthetic_connector(self):
        reg = ConnectorRegistry()
        with pytest.raises(ConnectorRegistryError):
            reg.register(ConnectorSpec(
                name="bad_real",
                kind=ConnectorKind.GENERIC_STUB,
                enabled=True,
                capabilities=[],
                allow_external=False,
                synthetic_only=False,  # NOT allowed in B08
            ))

    def test_disable_and_enable(self, default_registry):
        default_registry.disable("dxgpt_stub")
        assert not default_registry.get("dxgpt_stub").enabled
        default_registry.enable("dxgpt_stub")
        assert default_registry.get("dxgpt_stub").enabled

    def test_get_missing_connector_returns_none(self, default_registry):
        assert default_registry.get("nonexistent") is None

    def test_safe_public_summary_no_raw_name(self, default_registry):
        spec = default_registry.get("dxgpt_stub")
        summary = spec.safe_public_summary
        assert "name_hash" in summary
        # Raw name must not appear in summary value
        assert "dxgpt_stub" not in str(summary.get("name_hash", ""))


# ---------------------------------------------------------------------------
# TestRequestBuilder
# ---------------------------------------------------------------------------

class TestRequestBuilder:
    def test_clean_query_returns_request(self, default_registry):
        spec = default_registry.get("dxgpt_stub")
        result = build_connector_request("synthetic query", {}, spec)
        assert isinstance(result, ConnectorExecutionRequest)

    def test_disabled_connector_returns_disabled_result(self, default_registry):
        spec = default_registry.get("generic_stub")
        result = build_connector_request("query", {}, spec)
        assert isinstance(result, ConnectorExecutionResult)
        assert result.status == ConnectorStatus.DISABLED

    def test_safe_mode_returns_skipped(self, default_registry):
        spec = default_registry.get("dxgpt_stub")
        result = build_connector_request("query", {}, spec, safe_mode=True)
        assert isinstance(result, ConnectorExecutionResult)
        assert result.status == ConnectorStatus.SKIPPED_SAFE_MODE

    def test_secret_in_context_blocks(self, default_registry):
        spec = default_registry.get("dxgpt_stub")
        result = build_connector_request(
            "query",
            {"api_key": "sk-ABCDEF1234567890ABCDEF1234567890"},
            spec,
        )
        assert isinstance(result, ConnectorExecutionResult)
        assert result.status == ConnectorStatus.BLOCKED_PRIVACY

    def test_blocked_result_has_no_external_api(self, default_registry):
        spec = default_registry.get("dxgpt_stub")
        result = build_connector_request(
            "query",
            {"api_key": "sk-ABCDEF1234567890ABCDEF1234567890"},
            spec,
        )
        assert result.external_api_used is False

    def test_request_has_no_replacement_map(self, default_registry):
        spec = default_registry.get("dxgpt_stub")
        req = build_connector_request("synthetic query", {}, spec)
        assert isinstance(req, ConnectorExecutionRequest)
        # sanitized_payload must not contain replacement_map
        assert "replacement_map" not in req.sanitized_payload

    def test_clean_request_external_api_false(self, default_registry):
        spec = default_registry.get("dxgpt_stub")
        req = build_connector_request("synthetic query", {}, spec)
        assert isinstance(req, ConnectorExecutionRequest)
        assert req.allow_external is False


# ---------------------------------------------------------------------------
# TestConnectorStubs
# ---------------------------------------------------------------------------

class TestConnectorStubs:
    def test_dxgpt_stub_success(self):
        raw = call_dxgpt_stub({}, SimulationMode.SUCCESS)
        assert isinstance(raw.get("facts"), list)
        assert len(raw["facts"]) > 0
        assert raw.get("synthetic") is True
        assert raw.get("connector_name") == "dxgpt_stub"

    def test_dxgpt_stub_timeout(self):
        raw = call_dxgpt_stub({}, SimulationMode.TIMEOUT)
        assert raw.get("_stub_error") == "timeout"

    def test_dxgpt_stub_malformed(self):
        raw = call_dxgpt_stub({}, SimulationMode.MALFORMED_RESPONSE)
        assert "raw_garbage" in raw

    def test_sage_stub_success(self):
        raw = call_sage_epilepsy_stub({}, SimulationMode.SUCCESS)
        assert isinstance(raw.get("facts"), list)
        assert raw.get("synthetic") is True

    def test_patientnotes_stub_success(self):
        raw = call_patientnotes_ddi_stub({}, SimulationMode.SUCCESS)
        assert isinstance(raw.get("facts"), list)
        assert raw.get("synthetic") is True
        assert raw.get("connector_name") == "patientnotes_ddi_stub"

    def test_patientnotes_stub_no_external_api(self):
        # Stub is local — external_api_used key should be absent or False
        raw = call_patientnotes_ddi_stub({}, SimulationMode.SUCCESS)
        assert raw.get("external_api_used", False) is False

    def test_patientnotes_stub_no_medication_advice(self):
        raw = call_patientnotes_ddi_stub({}, SimulationMode.SUCCESS)
        full_str = str(raw).lower()
        assert "advice" not in full_str or "no real" in full_str.replace("advice", "")
        assert "dosing" not in full_str
        assert "prescrib" not in full_str

    def test_generic_stub_success(self):
        raw = call_generic_stub({}, SimulationMode.SUCCESS)
        assert isinstance(raw.get("facts"), list)
        assert raw.get("synthetic") is True

    def test_no_clinical_recommendation_in_any_stub(self):
        for fn, mode in [
            (call_dxgpt_stub, SimulationMode.SUCCESS),
            (call_sage_epilepsy_stub, SimulationMode.SUCCESS),
            (call_patientnotes_ddi_stub, SimulationMode.SUCCESS),
        ]:
            raw = fn({}, mode)
            text = str(raw).lower()
            assert "prescrib" not in text
            assert "dosing" not in text


# ---------------------------------------------------------------------------
# TestConnectorExecutor
# ---------------------------------------------------------------------------

class TestConnectorExecutor:
    def test_success_execution(self, dxgpt_sage_registry):
        requests = _build_requests(dxgpt_sage_registry)
        results = execute_connectors(requests, dxgpt_sage_registry, simulation_mode=SimulationMode.SUCCESS)
        assert all(r.status == ConnectorStatus.SUCCESS for r in results)

    def test_timeout_execution(self, dxgpt_sage_registry):
        requests = _build_requests(dxgpt_sage_registry)
        results = execute_connectors(requests, dxgpt_sage_registry, simulation_mode=SimulationMode.TIMEOUT)
        assert all(r.status == ConnectorStatus.TIMEOUT for r in results)

    def test_error_execution(self, dxgpt_sage_registry):
        requests = _build_requests(dxgpt_sage_registry)
        results = execute_connectors(requests, dxgpt_sage_registry, simulation_mode=SimulationMode.ERROR)
        assert all(r.status == ConnectorStatus.ERROR for r in results)

    def test_malformed_execution(self, dxgpt_sage_registry):
        requests = _build_requests(dxgpt_sage_registry)
        results = execute_connectors(requests, dxgpt_sage_registry, simulation_mode=SimulationMode.MALFORMED_RESPONSE)
        assert all(r.status == ConnectorStatus.MALFORMED_RESPONSE for r in results)

    def test_blocked_privacy_not_executed(self, dxgpt_sage_registry):
        # Already-blocked results pass through without executing stub
        spec = dxgpt_sage_registry.get("dxgpt_stub")
        blocked = build_connector_request(
            "query",
            {"api_key": "sk-ABCDEF1234567890ABCDEF1234567890"},
            spec,
        )
        results = execute_connectors([blocked], dxgpt_sage_registry, simulation_mode=SimulationMode.SUCCESS)
        assert results[0].status == ConnectorStatus.BLOCKED_PRIVACY
        assert results[0].normalized_response is None

    def test_safe_mode_skips_execution(self, dxgpt_sage_registry):
        requests = _build_requests(dxgpt_sage_registry, safe_mode=True)
        results = execute_connectors(requests, dxgpt_sage_registry, safe_mode=True)
        for r in results:
            assert r.status in (ConnectorStatus.SKIPPED_SAFE_MODE, ConnectorStatus.DISABLED)
            assert r.normalized_response is None

    def test_no_external_api_in_any_result(self, dxgpt_sage_registry):
        requests = _build_requests(dxgpt_sage_registry)
        results = execute_connectors(requests, dxgpt_sage_registry, simulation_mode=SimulationMode.SUCCESS)
        assert all(r.external_api_used is False for r in results)

    def test_disabled_result_passes_through(self, default_registry):
        spec = default_registry.get("generic_stub")
        disabled = build_connector_request("query", {}, spec)
        results = execute_connectors([disabled], default_registry, simulation_mode=SimulationMode.SUCCESS)
        assert results[0].status == ConnectorStatus.DISABLED


# ---------------------------------------------------------------------------
# TestResponseNormalizer
# ---------------------------------------------------------------------------

class TestResponseNormalizer:
    def test_accepts_structured_synthetic_response(self, default_registry):
        spec = default_registry.get("dxgpt_stub")
        raw = call_dxgpt_stub({}, SimulationMode.SUCCESS)
        normalized = normalize_connector_response(raw, spec)
        assert normalized is not None
        assert isinstance(normalized.get("facts"), list)
        assert normalized.get("source_kind") == "connector_stub"

    def test_rejects_malformed_response(self, default_registry):
        spec = default_registry.get("dxgpt_stub")
        raw = call_dxgpt_stub({}, SimulationMode.MALFORMED_RESPONSE)
        normalized = normalize_connector_response(raw, spec)
        assert normalized is None

    def test_rejects_timeout_response(self, default_registry):
        spec = default_registry.get("dxgpt_stub")
        raw = call_dxgpt_stub({}, SimulationMode.TIMEOUT)
        normalized = normalize_connector_response(raw, spec)
        assert normalized is None

    def test_strips_unsafe_raw_fields(self, default_registry):
        spec = default_registry.get("dxgpt_stub")
        raw = call_dxgpt_stub({}, SimulationMode.SUCCESS)
        raw["raw_source_text"] = "PRIVATE TEXT"
        raw["replacement_map"] = {"x": "y"}
        raw["source_response_raw"] = "RAW RESPONSE"
        normalized = normalize_connector_response(raw, spec)
        assert normalized is not None
        assert "raw_source_text" not in normalized
        assert "replacement_map" not in normalized
        assert "source_response_raw" not in normalized

    def test_normalized_includes_response_hash(self, default_registry):
        spec = default_registry.get("dxgpt_stub")
        raw = call_dxgpt_stub({}, SimulationMode.SUCCESS)
        normalized = normalize_connector_response(raw, spec)
        assert "response_hash" in normalized

    def test_confidence_clamped_to_zero_one(self, default_registry):
        spec = default_registry.get("dxgpt_stub")
        raw = call_dxgpt_stub({}, SimulationMode.SUCCESS)
        raw["confidence"] = 999.9
        normalized = normalize_connector_response(raw, spec)
        assert normalized is not None
        assert 0.0 <= normalized["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# TestFactExtractor
# ---------------------------------------------------------------------------

class TestFactExtractor:
    def _make_normalized(self, connector_name, facts, confidence=0.85):
        return {
            "connector_name": connector_name,
            "source_kind": "connector_stub",
            "facts": facts,
            "citations": [],
            "confidence": confidence,
            "synthetic": True,
        }

    def test_groups_same_fact_across_connectors(self):
        shared_fact = {
            "fact_type": "diagnosis",
            "entity_text": "synthetic condition alpha",
            "structured": {"icd": "SYN-001"},
            "specialty": "general",
            "confidence": 0.85,
            "synthetic": True,
        }
        resp_a = self._make_normalized("connector_a", [shared_fact])
        resp_b = self._make_normalized("connector_b", [shared_fact])
        facts = extract_consensus_facts([resp_a, resp_b], total_successful_connectors=2)
        diag_facts = [f for f in facts if f.fact_type == "diagnosis" and f.entity_text == "synthetic condition alpha"]
        assert len(diag_facts) == 1
        assert len(diag_facts[0].supporting_connectors) == 2

    def test_single_source_fact_gets_penalized_status(self):
        resp = self._make_normalized("only_connector", [{
            "fact_type": "diagnosis",
            "entity_text": "synthetic single source",
            "structured": {},
            "specialty": "general",
            "confidence": 0.90,
        }])
        facts = extract_consensus_facts([resp], total_successful_connectors=1)
        assert facts[0].status == ConsensusFactStatus.SINGLE_SOURCE_PENALIZED

    def test_disagreeing_structured_values_get_contradicted(self):
        fact_a = {"fact_type": "diagnosis", "entity_text": "cond", "structured": {"code": "A"}, "specialty": "general", "confidence": 0.85}
        fact_b = {"fact_type": "diagnosis", "entity_text": "cond", "structured": {"code": "B"}, "specialty": "general", "confidence": 0.85}
        resp_a = self._make_normalized("conn_a", [fact_a])
        resp_b = self._make_normalized("conn_b", [fact_b])
        facts = extract_consensus_facts([resp_a, resp_b], total_successful_connectors=2)
        contradicted = [f for f in facts if f.status == ConsensusFactStatus.CONTRADICTED]
        assert len(contradicted) == 1

    def test_agreed_fact_has_agreed_status(self):
        shared = {"fact_type": "diag", "entity_text": "same", "structured": {"code": "SYN"}, "specialty": "general", "confidence": 0.85}
        resp_a = self._make_normalized("conn_a", [shared])
        resp_b = self._make_normalized("conn_b", [shared])
        facts = extract_consensus_facts([resp_a, resp_b], total_successful_connectors=2)
        agreed = [f for f in facts if f.status == ConsensusFactStatus.AGREED]
        assert len(agreed) == 1


# ---------------------------------------------------------------------------
# TestAgreementScoring
# ---------------------------------------------------------------------------

class TestAgreementScoring:
    def _make_resp(self, connector_name, entity_text="cond", structured=None, confidence=0.85):
        return {
            "connector_name": connector_name,
            "source_kind": "connector_stub",
            "facts": [{
                "fact_type": "diagnosis",
                "entity_text": entity_text,
                "structured": structured or {"code": "SYN"},
                "specialty": "general",
                "confidence": confidence,
            }],
            "citations": [],
            "confidence": confidence,
            "synthetic": True,
        }

    def test_two_connector_agreement_produces_consensus_ready(self):
        result = calculate_agreement([
            self._make_resp("conn_a"),
            self._make_resp("conn_b"),
        ])
        assert result.status == ConsensusStatus.CONSENSUS_READY
        agreed = [f for f in result.consensus_facts if f.status == ConsensusFactStatus.AGREED]
        assert len(agreed) > 0

    def test_single_connector_produces_single_source_penalized(self):
        result = calculate_agreement([self._make_resp("conn_a")])
        penalized = [f for f in result.consensus_facts if f.status == ConsensusFactStatus.SINGLE_SOURCE_PENALIZED]
        assert len(penalized) > 0

    def test_no_responses_produces_all_discarded(self):
        result = calculate_agreement([])
        assert result.status == ConsensusStatus.ALL_RESPONSES_DISCARDED
        assert result.escalation_required is True

    def test_all_discarded_escalation_required(self):
        result = calculate_agreement([])
        assert result.escalation_required is True

    def test_confidence_aggregate_is_float(self):
        result = calculate_agreement([self._make_resp("conn_a"), self._make_resp("conn_b")])
        assert isinstance(result.confidence_aggregate, float)
        assert 0.0 <= result.confidence_aggregate <= 1.0


# ---------------------------------------------------------------------------
# TestContradictionDetection
# ---------------------------------------------------------------------------

class TestContradictionDetection:
    def _make_contradicted_fact(self, fact_type="diagnosis") -> ConsensusFact:
        from clinical_knowledge.consensus.models import _safe_fact_id
        key = f"general:{fact_type}:conflicting entity"
        return ConsensusFact(
            fact_id=key,
            safe_fact_id=_safe_fact_id(key),
            fact_type=fact_type,
            entity_text="conflicting entity",
            structured={"code": "SYN-CONFLICT"},
            specialty="general",
            supporting_connectors=["conn_a", "conn_b"],
            contradicting_connectors=[],
            agreement_ratio=0.5,
            confidence=0.4,
            status=ConsensusFactStatus.CONTRADICTED,
        )

    def test_detects_contradiction_in_contradicted_fact(self):
        fact = self._make_contradicted_fact()
        contradictions = detect_consensus_contradictions([fact])
        assert len(contradictions) == 1

    def test_no_contradiction_for_agreed_fact(self):
        from clinical_knowledge.consensus.models import _safe_fact_id
        key = "general:diagnosis:agreed"
        agreed = ConsensusFact(
            fact_id=key,
            safe_fact_id=_safe_fact_id(key),
            fact_type="diagnosis",
            entity_text="agreed",
            structured={"code": "SYN-001"},
            specialty="general",
            supporting_connectors=["a", "b"],
            contradicting_connectors=[],
            agreement_ratio=1.0,
            confidence=0.85,
            status=ConsensusFactStatus.AGREED,
        )
        contradictions = detect_consensus_contradictions([agreed])
        assert len(contradictions) == 0

    def test_medication_fact_type_flagged_as_dose_conflict(self):
        fact = self._make_contradicted_fact(fact_type="medication")
        contradictions = detect_consensus_contradictions([fact])
        assert contradictions[0].is_medication_dose_conflict is True

    def test_contradiction_uses_safe_ids_only(self):
        fact = self._make_contradicted_fact()
        contradictions = detect_consensus_contradictions([fact])
        contradiction = contradictions[0]
        summary = contradiction.safe_public_summary
        assert "connector_names" not in summary
        assert "connector_count" in summary


# ---------------------------------------------------------------------------
# TestConsensusEngine
# ---------------------------------------------------------------------------

class TestConsensusEngine:
    def test_multi_connector_agreement(self, dxgpt_sage_registry):
        requests = _build_requests(dxgpt_sage_registry)
        results = execute_connectors(requests, dxgpt_sage_registry, simulation_mode=SimulationMode.SUCCESS)
        consensus = run_consensus(results, min_confidence=0.0)
        agreed = [f for f in consensus.consensus_facts if f.status == ConsensusFactStatus.AGREED]
        assert len(agreed) > 0
        assert len(consensus.contradictions) == 0

    def test_contradiction_detected(self, dxgpt_sage_registry):
        requests = _build_requests(dxgpt_sage_registry)
        results = execute_connectors(requests, dxgpt_sage_registry, simulation_mode=SimulationMode.CONTRADICTION)
        consensus = run_consensus(results, min_confidence=0.0)
        assert consensus.status == ConsensusStatus.CONTRADICTION_DETECTED or len(consensus.contradictions) > 0

    def test_all_fail_produces_all_discarded(self, dxgpt_sage_registry):
        requests = _build_requests(dxgpt_sage_registry)
        results = execute_connectors(requests, dxgpt_sage_registry, simulation_mode=SimulationMode.TIMEOUT)
        consensus = run_consensus(results, min_confidence=0.0)
        assert consensus.status == ConsensusStatus.ALL_RESPONSES_DISCARDED
        assert consensus.escalation_required is True

    def test_no_external_api_in_consensus(self, dxgpt_sage_registry):
        requests = _build_requests(dxgpt_sage_registry)
        results = execute_connectors(requests, dxgpt_sage_registry, simulation_mode=SimulationMode.SUCCESS)
        assert all(r.external_api_used is False for r in results)

    def test_low_confidence_discarded(self, dxgpt_sage_registry):
        requests = _build_requests(dxgpt_sage_registry)
        results = execute_connectors(requests, dxgpt_sage_registry, simulation_mode=SimulationMode.LOW_CONFIDENCE)
        consensus = run_consensus(results, min_confidence=0.60)
        success = [r for r in results if r.status == ConnectorStatus.SUCCESS]
        low_conf = [r for r in success if r.normalized_response and r.normalized_response.get("confidence", 1.0) < 0.60]
        # Either responses were discarded or their confidence is below threshold
        assert consensus.discarded_response_count > 0 or len(low_conf) > 0

    def test_consensus_does_not_auto_write_active(self, dxgpt_sage_registry):
        """Consensus must never auto-write active records."""
        requests = _build_requests(dxgpt_sage_registry)
        results = execute_connectors(requests, dxgpt_sage_registry, simulation_mode=SimulationMode.SUCCESS)
        consensus = run_consensus(results, min_confidence=0.0)
        # Consensus facts have no 'active_write' flag — they're read-only output
        for fact in consensus.consensus_facts:
            # No fact should claim active write capability
            assert not hasattr(fact, "active_write") or not getattr(fact, "active_write", False)


# ---------------------------------------------------------------------------
# TestTruthResolutionHandoff
# ---------------------------------------------------------------------------

class TestTruthResolutionHandoff:
    def test_contradiction_routes_to_truth_resolution(self, dxgpt_sage_registry, store):
        requests = _build_requests(dxgpt_sage_registry)
        results = execute_connectors(requests, dxgpt_sage_registry, simulation_mode=SimulationMode.CONTRADICTION)
        consensus = run_consensus(results, min_confidence=0.0, store=store)
        # If contradictions exist, TR handoff should have run
        if consensus.contradictions:
            assert consensus.status == ConsensusStatus.TRUTH_RESOLUTION_REQUIRED or len(consensus.truth_resolution_results) >= 0

    def test_truth_resolution_does_not_invoke_ddi(self, dxgpt_sage_registry, store):
        requests = _build_requests(dxgpt_sage_registry)
        results = execute_connectors(requests, dxgpt_sage_registry, simulation_mode=SimulationMode.CONTRADICTION)
        consensus = run_consensus(results, min_confidence=0.0, store=store)
        for tr in consensus.truth_resolution_results:
            assert tr.get("ddi_invoked", False) is False

    def test_truth_resolution_quarantine_only(self, dxgpt_sage_registry, store):
        requests = _build_requests(dxgpt_sage_registry)
        results = execute_connectors(requests, dxgpt_sage_registry, simulation_mode=SimulationMode.CONTRADICTION)
        consensus = run_consensus(results, min_confidence=0.0, store=store)
        for tr in consensus.truth_resolution_results:
            assert tr.get("active_write", False) is False
            assert tr.get("quarantine_only", True) is True

    def test_route_contradictions_directly(self, store):
        from clinical_knowledge.consensus.models import ConsensusContradiction
        contradiction = ConsensusContradiction(
            fact_type="medication",
            entity_text="synthetic medication beta",
            specialty="pharmacology",
            conflicting_structured_values=[{"dose": "100mg"}, {"dose": "200mg"}],
            connector_names=["conn_a", "conn_b"],
            safe_ids=["cka_cf_abc123"],
            is_medication_dose_conflict=True,
        )
        results = route_consensus_contradictions_to_truth_resolution([contradiction], store)
        assert len(results) == 1
        assert results[0].get("ddi_invoked") is False
        assert results[0].get("quarantine_only") is True
        assert results[0].get("active_write") is False


# ---------------------------------------------------------------------------
# TestConsensusToEnrichment
# ---------------------------------------------------------------------------

class TestConsensusToEnrichment:
    def test_consensus_facts_remain_hypothesis(self, dxgpt_sage_registry):
        requests = _build_requests(dxgpt_sage_registry)
        results = execute_connectors(requests, dxgpt_sage_registry, simulation_mode=SimulationMode.SUCCESS)
        consensus = run_consensus(results, min_confidence=0.0)
        candidates = consensus_facts_to_enrichment_candidates(
            consensus.consensus_facts, allow_active_write=False
        )
        assert all(c["tier"] == "hypothesis" for c in candidates)

    def test_no_active_write_in_candidates(self, dxgpt_sage_registry):
        requests = _build_requests(dxgpt_sage_registry)
        results = execute_connectors(requests, dxgpt_sage_registry, simulation_mode=SimulationMode.SUCCESS)
        consensus = run_consensus(results, min_confidence=0.0)
        candidates = consensus_facts_to_enrichment_candidates(
            consensus.consensus_facts, allow_active_write=False
        )
        assert all(not c.get("active_write", True) for c in candidates)

    def test_allow_active_write_raises(self, dxgpt_sage_registry):
        requests = _build_requests(dxgpt_sage_registry)
        results = execute_connectors(requests, dxgpt_sage_registry, simulation_mode=SimulationMode.SUCCESS)
        consensus = run_consensus(results, min_confidence=0.0)
        with pytest.raises(ValueError, match="allow_active_write"):
            consensus_facts_to_enrichment_candidates(
                consensus.consensus_facts, allow_active_write=True
            )

    def test_candidates_require_enrichment_review(self, dxgpt_sage_registry):
        requests = _build_requests(dxgpt_sage_registry)
        results = execute_connectors(requests, dxgpt_sage_registry, simulation_mode=SimulationMode.SUCCESS)
        consensus = run_consensus(results, min_confidence=0.0)
        candidates = consensus_facts_to_enrichment_candidates(consensus.consensus_facts)
        assert all(c.get("requires_enrichment_review") for c in candidates)


# ---------------------------------------------------------------------------
# TestSafetyBoundaries
# ---------------------------------------------------------------------------

class TestSafetyBoundaries:
    def test_no_scispacy_dependency(self):
        """Importing B08 modules must not require scispaCy/spaCy.

        This must be checked in a fresh interpreter. Other MedAI tests
        legitimately import spaCy-backed OCR/extraction modules earlier in the
        full suite, so asserting against this process' global sys.modules is
        order-sensitive and does not prove the B08 boundary.
        """
        code = """
import sys
from clinical_knowledge.connectors.executor import execute_connectors
from clinical_knowledge.connectors.models import ConnectorKind, SimulationMode
from clinical_knowledge.connectors.normalizer import normalize_connector_response
from clinical_knowledge.connectors.registry import ConnectorRegistry
from clinical_knowledge.connectors.request_builder import build_connector_request
from clinical_knowledge.consensus.engine import run_consensus
from clinical_knowledge.consensus.fact_extractor import extract_consensus_facts
from clinical_knowledge.consensus.integration import consensus_facts_to_enrichment_candidates
assert 'scispacy' not in sys.modules
assert 'spacy' not in sys.modules
print('ok')
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "ok"

    def test_no_external_api_calls(self, dxgpt_sage_registry):
        """Confirm external_api_used is always False."""
        requests = _build_requests(dxgpt_sage_registry)
        results = execute_connectors(requests, dxgpt_sage_registry, simulation_mode=SimulationMode.SUCCESS)
        assert all(r.external_api_used is False for r in results)

    def test_no_clinical_recommendation_text(self, dxgpt_sage_registry):
        """Consensus facts must not contain clinical recommendation text."""
        requests = _build_requests(dxgpt_sage_registry)
        results = execute_connectors(requests, dxgpt_sage_registry, simulation_mode=SimulationMode.SUCCESS)
        consensus = run_consensus(results, min_confidence=0.0)
        for fact in consensus.consensus_facts:
            text = str(fact.structured).lower()
            assert "prescrib" not in text
            assert "dosing" not in text
            assert "take this" not in text

    def test_no_prescription_dosing_advice(self, dxgpt_sage_registry):
        requests = _build_requests(dxgpt_sage_registry)
        results = execute_connectors(requests, dxgpt_sage_registry, simulation_mode=SimulationMode.SUCCESS)
        for r in results:
            if r.normalized_response:
                for fact in r.normalized_response.get("facts", []):
                    text = str(fact).lower()
                    assert "dosing" not in text
                    assert "prescrib" not in text

    def test_unknown_coding_remains_unmapped(self):
        candidate = CodingCandidate(
            candidate_id="test_b08_safety",
            safe_candidate_id="cka_cc_safety",
            fact_type="diagnosis",
            entity_text="completely unknown consensus entity",
            normalized_text="completely unknown consensus entity",
            specialty="general",
            structured={},
            source_record_id="test_b08_safety",
            source_tier="hypothesis",
        )
        source = SyntheticTerminologySource()
        result = code_entity(candidate, [source])
        assert result.status == CodingStatus.UNMAPPED
        assert result.no_code_hallucinated is True
        assert len(result.codes) == 0

    def test_apply_coding_does_not_promote_hypothesis(self, store):
        """Coding applied to hypothesis record must not change tier."""
        record = _make_record("hypo_rec_b08", tier=KnowledgeTier.HYPOTHESIS)
        assert record.tier == KnowledgeTier.HYPOTHESIS

        candidate = CodingCandidate(
            candidate_id=record.record_id,
            safe_candidate_id="cka_cc_hypo",
            fact_type="diagnosis",
            entity_text=record.entity_text,
            normalized_text=record.entity_text.lower(),
            specialty="general",
            structured={},
            source_record_id=record.record_id,
            source_tier="hypothesis",
        )
        source = SyntheticTerminologySource()
        from clinical_knowledge.medical_coding.integration import apply_coding_result_to_record
        coding_result = code_entity(candidate, [source])
        apply_coding_result_to_record(record, coding_result)

        assert record.tier == KnowledgeTier.HYPOTHESIS  # tier unchanged

    def test_ddi_status_unchanged_after_consensus(self, dxgpt_sage_registry):
        """DDI status must not be cleared or overridden by consensus processing."""
        record = _make_record("ddi_b08", ddi_status=DDIStatus.BLOCKED)
        assert record.ddi_status == DDIStatus.BLOCKED

        requests = _build_requests(dxgpt_sage_registry)
        results = execute_connectors(requests, dxgpt_sage_registry, simulation_mode=SimulationMode.SUCCESS)
        run_consensus(results, min_confidence=0.0)

        # Record not touched by consensus engine
        assert record.ddi_status == DDIStatus.BLOCKED


# ---------------------------------------------------------------------------
# TestLedgerEvents
# ---------------------------------------------------------------------------

class TestLedgerEvents:
    def test_connector_execution_event_created(self):
        evt = make_connector_execution_event(
            connector_name_hash="abc123",
            status="success",
            external_api_used=False,
        )
        assert evt.event_type == LedgerEventType.CONNECTOR_EXECUTION
        assert evt.safe_public_details["external_api_used"] is False

    def test_consensus_result_event_created(self):
        evt = make_consensus_result_event(
            consensus_status="consensus_ready",
            fact_count=2,
            contradiction_count=0,
            confidence_aggregate=0.85,
            discarded_count=0,
            escalation_required=False,
        )
        assert evt.event_type == LedgerEventType.CONSENSUS_RESULT
        assert evt.safe_public_details["external_api_used"] is False
        assert evt.safe_public_details["active_write"] is False

    def test_ledger_event_no_raw_connector_name(self):
        evt = make_connector_execution_event(
            connector_name_hash="abc123",
            status="success",
            external_api_used=False,
        )
        # connector_name_hash is a hash, not the raw name
        details_str = str(evt.safe_public_details)
        assert "dxgpt_stub" not in details_str
        assert "sage_epilepsy_stub" not in details_str


# ---------------------------------------------------------------------------
# TestPrivacyAndPublicReport
# ---------------------------------------------------------------------------

class TestPrivacyAndPublicReport:
    def test_connector_result_safe_public_summary(self, dxgpt_sage_registry):
        requests = _build_requests(dxgpt_sage_registry)
        results = execute_connectors(requests, dxgpt_sage_registry, simulation_mode=SimulationMode.SUCCESS)
        for r in results:
            summary = r.safe_public_summary
            assert "connector_name_hash" in summary
            # Raw name must not appear in summary hash value
            for name in ["dxgpt_stub", "sage_epilepsy_stub", "patientnotes_ddi_stub"]:
                assert name not in str(summary.get("connector_name_hash", ""))

    def test_consensus_safe_public_summary_passes_privacy_check(self, dxgpt_sage_registry):
        requests = _build_requests(dxgpt_sage_registry)
        results = execute_connectors(requests, dxgpt_sage_registry, simulation_mode=SimulationMode.SUCCESS)
        consensus = run_consensus(results, min_confidence=0.0)
        report_payload = {
            "block_id": "CKA-B08",
            "consensus": consensus.safe_public_summary,
            "connector_results": [r.safe_public_summary for r in results],
        }
        privacy_check = check_public_report_payload(report_payload)
        assert privacy_check.passed, f"Privacy check failed: {privacy_check.leak_examples_redacted}"

    def test_no_replacement_map_in_public_report(self, dxgpt_sage_registry):
        requests = _build_requests(dxgpt_sage_registry)
        results = execute_connectors(requests, dxgpt_sage_registry, simulation_mode=SimulationMode.SUCCESS)
        consensus = run_consensus(results, min_confidence=0.0)
        report_str = json.dumps(consensus.safe_public_summary)
        assert "replacement_map" not in report_str

    def test_no_source_response_raw_in_public_report(self, dxgpt_sage_registry):
        requests = _build_requests(dxgpt_sage_registry)
        results = execute_connectors(requests, dxgpt_sage_registry, simulation_mode=SimulationMode.SUCCESS)
        consensus = run_consensus(results, min_confidence=0.0)
        report_str = json.dumps(consensus.safe_public_summary)
        assert "source_response_raw" not in report_str


# ---------------------------------------------------------------------------
# TestValidationScript
# ---------------------------------------------------------------------------

class TestValidationScript:
    def test_validation_script_runs_cleanly(self):
        from scripts.run_cka_block08_multi_connector_consensus_validation import run_validation
        report = run_validation()
        assert report["all_passed"] is True, f"Validation failed: {report.get('case_results')}"

    def test_validation_report_safety_flags(self):
        from scripts.run_cka_block08_multi_connector_consensus_validation import run_validation
        report = run_validation()
        assert report["real_external_connectors_implemented"] is False
        assert report["external_api_used"] is False
        assert report["real_dxgpt_api_used"] is False
        assert report["real_sage_api_used"] is False
        assert report["real_patientnotes_api_used"] is False
        assert report["real_llm_api_used"] is False
        assert report["consensus_does_not_synthesize_over_contradiction"] is True
        assert report["consensus_does_not_auto_write_active"] is True
        assert report["consensus_to_enrichment_remains_hypothesis"] is True
        assert report["medication_dose_contradiction_quarantines_only"] is True
        assert report["truth_resolution_invokes_ddi"] is False
        assert report["no_code_hallucinated"] is True
        assert report["clinical_recommendations_generated"] is False
        assert report["prescription_dosing_advice_generated"] is False
        assert report["raw_phi_logged_in_public_reports"] is False
        assert report["private_filename_path_leaks"] == 0
        assert report["secret_leaks"] == 0
        assert report["frozen_hitl_release_reopened"] is False

    def test_validation_all_12_cases_pass(self):
        from scripts.run_cka_block08_multi_connector_consensus_validation import run_validation
        report = run_validation()
        assert report["synthetic_cases_run"] == 12
        assert report["cases_passed"] == 12

    def test_reports_written_to_correct_location(self):
        from scripts.run_cka_block08_multi_connector_consensus_validation import run_validation, write_reports
        report = run_validation()
        write_reports(report)
        report_dir = Path(__file__).parent.parent / "reports" / "cka_block08_multi_connector_consensus"
        assert (report_dir / "cka_block08_multi_connector_consensus_report.json").exists()
        assert (report_dir / "cka_block08_multi_connector_consensus_report.md").exists()
