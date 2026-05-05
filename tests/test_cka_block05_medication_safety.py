"""Tests for CKA-B05 Medication Safety / DDI Dual-Layer Gate."""
from __future__ import annotations

import pytest

from clinical_knowledge.medication_safety.ddi_stub import check_ddi_stub
from clinical_knowledge.medication_safety.evidence_modifier import apply_ddi_evidence_modifier
from clinical_knowledge.medication_safety.integration import attempt_medication_record_write
from clinical_knowledge.medication_safety.models import (
    DDICheckResult,
    DDICheckStatus,
    DDIFinding,
    DDISeverity,
    Layer1DDIScoreResult,
    MedicationSafetyAction,
    MedicationWriteGateResult,
)
from clinical_knowledge.medication_safety.write_gate import evaluate_medication_write_gate
from clinical_knowledge.models import (
    KnowledgeTier, LedgerEventType, MKBRecord, RecordStatus, SourceType, TrustLevel,
)
from clinical_knowledge.safe_ids import make_safe_record_id, new_record_id
from clinical_knowledge.store import MKBStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _med_rec(entity="synth_med_clear", fact_type="medication") -> MKBRecord:
    rid = new_record_id()
    return MKBRecord(
        record_id=rid,
        safe_record_id=make_safe_record_id(rid),
        session_id="test_b05",
        fact_type=fact_type,
        entity_text=entity,
        trust_level=TrustLevel.UNVERIFIED,
        tier=KnowledgeTier.ACTIVE,
        status=RecordStatus.CONFIRMED,
        source_type=SourceType.SYNTHETIC,
        confidence=0.80,
    )


@pytest.fixture
def empty_store():
    return MKBStore(db_path=":memory:")


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestEnums:
    def test_ddi_severity_values(self):
        assert DDISeverity.NONE == "none"
        assert DDISeverity.LOW == "low"
        assert DDISeverity.MEDIUM == "medium"
        assert DDISeverity.HIGH == "high"
        assert DDISeverity.UNAVAILABLE == "unavailable"

    def test_ddi_check_status_values(self):
        assert DDICheckStatus.CLEAR == "clear"
        assert DDICheckStatus.LOW == "low"
        assert DDICheckStatus.MEDIUM == "medium"
        assert DDICheckStatus.HIGH_BLOCKED == "high_blocked"
        assert DDICheckStatus.PENDING == "pending"
        assert DDICheckStatus.UNAVAILABLE == "unavailable"

    def test_medication_safety_action_values(self):
        assert MedicationSafetyAction.ALLOW == "allow"
        assert MedicationSafetyAction.ALLOW_WITH_NOTE == "allow_with_note"
        assert MedicationSafetyAction.WARN_REQUIRES_ACK == "warn_requires_ack"
        assert MedicationSafetyAction.BLOCK_REQUIRES_CONFIRMATION == "block_requires_confirmation"
        assert MedicationSafetyAction.QUEUE_PENDING_DDI == "queue_pending_ddi"

    def test_ddi_block_ddi_warning_now_active(self):
        from clinical_knowledge.models import _RESERVED_EVENT_TYPES
        assert LedgerEventType.DDI_BLOCK not in _RESERVED_EVENT_TYPES
        assert LedgerEventType.DDI_WARNING not in _RESERVED_EVENT_TYPES


# ---------------------------------------------------------------------------
# DDI stub tests
# ---------------------------------------------------------------------------


class TestDDIStub:
    def test_normal_mode_no_interaction(self):
        result = check_ddi_stub("synth_med_clear", ["other_synth"])
        assert result.checked is True
        assert result.available is True
        assert result.highest_severity == DDISeverity.NONE
        assert result.status == DDICheckStatus.CLEAR
        assert result.findings == []

    def test_normal_mode_high_interaction(self):
        result = check_ddi_stub("synth_med_alpha", ["synth_med_beta"])
        assert result.highest_severity == DDISeverity.HIGH
        assert result.status == DDICheckStatus.HIGH_BLOCKED
        assert len(result.findings) == 1

    def test_normal_mode_medium_interaction(self):
        result = check_ddi_stub("synth_med_gamma", ["synth_med_delta"])
        assert result.highest_severity == DDISeverity.MEDIUM
        assert result.status == DDICheckStatus.MEDIUM

    def test_normal_mode_low_interaction(self):
        result = check_ddi_stub("synth_med_epsilon", ["synth_med_zeta"])
        assert result.highest_severity == DDISeverity.LOW
        assert result.status == DDICheckStatus.LOW

    def test_force_none_mode(self):
        result = check_ddi_stub("synth_med_alpha", ["synth_med_beta"], mode="force_none")
        assert result.highest_severity == DDISeverity.NONE
        assert result.findings == []

    def test_force_high_mode(self):
        result = check_ddi_stub("any_med", [], mode="force_high")
        assert result.highest_severity == DDISeverity.HIGH
        assert result.status == DDICheckStatus.HIGH_BLOCKED

    def test_force_medium_mode(self):
        result = check_ddi_stub("any_med", [], mode="force_medium")
        assert result.highest_severity == DDISeverity.MEDIUM

    def test_force_low_mode(self):
        result = check_ddi_stub("any_med", [], mode="force_low")
        assert result.highest_severity == DDISeverity.LOW

    def test_unavailable_mode(self):
        result = check_ddi_stub("any_med", [], mode="unavailable")
        assert result.checked is False
        assert result.available is False
        assert result.highest_severity == DDISeverity.UNAVAILABLE
        assert result.status == DDICheckStatus.UNAVAILABLE

    def test_no_network_calls_required(self):
        # Calling stub should succeed without any network; if it reaches out it will fail
        import socket
        original = socket.socket
        def fail_socket(*args, **kwargs):
            raise RuntimeError("Network call attempted in DDI stub — not allowed")
        socket.socket = fail_socket
        try:
            result = check_ddi_stub("synth_med_alpha", ["synth_med_beta"])
            assert result is not None
        finally:
            socket.socket = original

    def test_stub_findings_are_synthetic(self):
        result = check_ddi_stub("synth_med_alpha", ["synth_med_beta"])
        for f in result.findings:
            assert f.synthetic is True

    def test_safe_public_summary_present(self):
        result = check_ddi_stub("synth_med_alpha", ["synth_med_beta"])
        assert isinstance(result.safe_public_summary, dict)
        assert "highest_severity" in result.safe_public_summary

    def test_safe_public_summary_no_raw_phi(self):
        result = check_ddi_stub("synth_med_alpha", ["synth_med_beta"])
        summary_str = str(result.safe_public_summary)
        assert "@" not in summary_str
        assert "DOB" not in summary_str.upper()
        assert "MRN" not in summary_str.upper()

    def test_finding_management_note_no_prescription_advice(self):
        result = check_ddi_stub("synth_med_alpha", ["synth_med_beta"])
        for f in result.findings:
            note_lower = f.management_note.lower()
            assert "prescribe" not in note_lower
            assert "take" not in note_lower
            assert "dose" not in note_lower
            assert "mg" not in note_lower

    def test_reversed_drug_order_detected(self):
        result_fwd = check_ddi_stub("synth_med_alpha", ["synth_med_beta"])
        result_rev = check_ddi_stub("synth_med_beta", ["synth_med_alpha"])
        assert result_fwd.highest_severity == DDISeverity.HIGH
        assert result_rev.highest_severity == DDISeverity.HIGH


# ---------------------------------------------------------------------------
# Layer 1 evidence modifier tests
# ---------------------------------------------------------------------------


class TestLayer1EvidenceModifier:
    def _ddi(self, mode):
        return check_ddi_stub("x", [], mode=mode)

    def test_high_penalty_040(self):
        result = apply_ddi_evidence_modifier(0.80, self._ddi("force_high"))
        assert abs(result.adjusted_score - 0.40) < 1e-6
        assert abs(result.penalty_applied - 0.40) < 1e-6

    def test_medium_penalty_020(self):
        result = apply_ddi_evidence_modifier(0.80, self._ddi("force_medium"))
        assert abs(result.adjusted_score - 0.60) < 1e-6

    def test_low_penalty_005(self):
        result = apply_ddi_evidence_modifier(0.80, self._ddi("force_low"))
        assert abs(result.adjusted_score - 0.75) < 1e-6

    def test_none_no_penalty(self):
        result = apply_ddi_evidence_modifier(0.80, self._ddi("force_none"))
        assert abs(result.adjusted_score - 0.80) < 1e-6
        assert abs(result.penalty_applied - 0.0) < 1e-6

    def test_unavailable_caps_at_050(self):
        result = apply_ddi_evidence_modifier(0.80, self._ddi("unavailable"))
        assert result.adjusted_score <= 0.50

    def test_unavailable_low_base_no_increase(self):
        result = apply_ddi_evidence_modifier(0.30, self._ddi("unavailable"))
        assert result.adjusted_score <= 0.30

    def test_high_penalty_floored_at_zero(self):
        result = apply_ddi_evidence_modifier(0.20, self._ddi("force_high"))
        assert result.adjusted_score == 0.0

    def test_layer1_does_not_block_writes(self):
        result = apply_ddi_evidence_modifier(0.80, self._ddi("force_high"))
        assert result.safe_public_summary["blocks_write"] is False

    def test_returns_layer1_result_type(self):
        result = apply_ddi_evidence_modifier(0.80, self._ddi("force_none"))
        assert isinstance(result, Layer1DDIScoreResult)


# ---------------------------------------------------------------------------
# Write gate tests
# ---------------------------------------------------------------------------


class TestWriteGate:
    def _cand(self, entity="synth_med_clear"):
        return _med_rec(entity)

    def test_high_blocks_without_confirmation(self):
        cand = self._cand()
        result = evaluate_medication_write_gate(cand, [], ddi_mode="force_high")
        assert result.allowed_to_write is False
        assert result.action == MedicationSafetyAction.BLOCK_REQUIRES_CONFIRMATION
        assert result.requires_user_confirmation is True
        assert result.ddi_status == DDICheckStatus.HIGH_BLOCKED

    def test_medium_requires_ack(self):
        cand = self._cand()
        result = evaluate_medication_write_gate(cand, [], ddi_mode="force_medium")
        assert result.allowed_to_write is False
        assert result.action == MedicationSafetyAction.WARN_REQUIRES_ACK
        assert result.requires_user_confirmation is True

    def test_low_allows_with_note(self):
        cand = self._cand()
        result = evaluate_medication_write_gate(cand, [], ddi_mode="force_low")
        assert result.allowed_to_write is True
        assert result.action == MedicationSafetyAction.ALLOW_WITH_NOTE
        assert result.requires_user_confirmation is False

    def test_none_allows(self):
        cand = self._cand()
        result = evaluate_medication_write_gate(cand, [], ddi_mode="force_none")
        assert result.allowed_to_write is True
        assert result.action == MedicationSafetyAction.ALLOW

    def test_unavailable_queues(self):
        cand = self._cand()
        result = evaluate_medication_write_gate(cand, [], ddi_mode="unavailable")
        assert result.allowed_to_write is False
        assert result.action == MedicationSafetyAction.QUEUE_PENDING_DDI
        assert result.ddi_checked is False
        assert result.ddi_status == DDICheckStatus.PENDING

    def test_high_explanation_no_prescription_advice(self):
        cand = self._cand()
        result = evaluate_medication_write_gate(cand, [], ddi_mode="force_high")
        expl = result.explanation.lower()
        assert "prescribe" not in expl
        assert "recommend" not in expl
        assert "take" not in expl
        assert "mg" not in expl

    def test_medium_explanation_no_medication_advice(self):
        cand = self._cand()
        result = evaluate_medication_write_gate(cand, [], ddi_mode="force_medium")
        assert "prescribe" not in result.explanation.lower()

    def test_safe_public_summary_no_raw_phi(self):
        cand = self._cand()
        result = evaluate_medication_write_gate(cand, [], ddi_mode="force_high")
        summary_str = str(result.safe_public_summary)
        assert "@" not in summary_str
        assert "DOB" not in summary_str.upper()

    def test_ledger_event_ready_high(self):
        cand = self._cand()
        result = evaluate_medication_write_gate(cand, [], ddi_mode="force_high")
        assert result.ledger_event_ready is True

    def test_ledger_event_ready_medium(self):
        cand = self._cand()
        result = evaluate_medication_write_gate(cand, [], ddi_mode="force_medium")
        assert result.ledger_event_ready is True

    def test_ledger_not_needed_for_low(self):
        cand = self._cand()
        result = evaluate_medication_write_gate(cand, [], ddi_mode="force_low")
        assert result.ledger_event_ready is False


# ---------------------------------------------------------------------------
# Store integration tests
# ---------------------------------------------------------------------------


class TestStoreIntegration:
    def test_none_allows_and_inserts(self, empty_store):
        cand = _med_rec()
        result = attempt_medication_record_write(
            cand, empty_store, ddi_mode="force_none"
        )
        assert result.allowed_to_write is True
        active = empty_store.list_active()
        assert any(r["record_id"] == cand.record_id for r in active)

    def test_low_allows_and_inserts(self, empty_store):
        cand = _med_rec()
        result = attempt_medication_record_write(
            cand, empty_store, ddi_mode="force_low"
        )
        assert result.allowed_to_write is True

    def test_medium_without_ack_does_not_insert(self, empty_store):
        cand = _med_rec()
        result = attempt_medication_record_write(
            cand, empty_store, ddi_mode="force_medium"
        )
        assert result.allowed_to_write is False
        active_ids = {r["record_id"] for r in empty_store.list_active()}
        assert cand.record_id not in active_ids

    def test_medium_with_ack_inserts_with_review(self, empty_store):
        cand = _med_rec()
        result = attempt_medication_record_write(
            cand, empty_store, ddi_mode="force_medium", user_acknowledged=True
        )
        assert result.allowed_to_write is True
        active = empty_store.list_active()
        written = next((r for r in active if r["record_id"] == cand.record_id), None)
        assert written is not None
        assert written["requires_review"] == 1

    def test_high_without_confirmation_does_not_insert(self, empty_store):
        cand = _med_rec()
        result = attempt_medication_record_write(
            cand, empty_store, ddi_mode="force_high"
        )
        assert result.allowed_to_write is False
        active_ids = {r["record_id"] for r in empty_store.list_active()}
        assert cand.record_id not in active_ids

    def test_high_with_confirmation_inserts_with_review(self, empty_store):
        cand = _med_rec()
        result = attempt_medication_record_write(
            cand, empty_store, ddi_mode="force_high", user_confirmed_high=True
        )
        assert result.allowed_to_write is True
        active = empty_store.list_active()
        written = next((r for r in active if r["record_id"] == cand.record_id), None)
        assert written is not None
        assert written["requires_review"] == 1

    def test_unavailable_does_not_insert(self, empty_store):
        cand = _med_rec()
        result = attempt_medication_record_write(
            cand, empty_store, ddi_mode="unavailable"
        )
        assert result.allowed_to_write is False
        active_ids = {r["record_id"] for r in empty_store.list_active()}
        assert cand.record_id not in active_ids

    def test_non_medication_bypasses_gate(self, empty_store):
        cand = _med_rec(fact_type="lab_value")
        result = attempt_medication_record_write(
            cand, empty_store, ddi_mode="force_high"
        )
        assert result.allowed_to_write is True
        assert result.ddi_checked is False
        active_ids = {r["record_id"] for r in empty_store.list_active()}
        assert cand.record_id in active_ids

    def test_ddi_block_ledger_event_written_for_high(self, empty_store):
        cand = _med_rec()
        attempt_medication_record_write(cand, empty_store, ddi_mode="force_high")
        events = empty_store.read_ledger_events()
        types = [e["event_type"] for e in events]
        assert LedgerEventType.DDI_BLOCK.value in types

    def test_ddi_warning_ledger_event_written_for_medium(self, empty_store):
        cand = _med_rec()
        attempt_medication_record_write(cand, empty_store, ddi_mode="force_medium")
        events = empty_store.read_ledger_events()
        types = [e["event_type"] for e in events]
        assert LedgerEventType.DDI_WARNING.value in types

    def test_ledger_event_no_raw_phi(self, empty_store):
        cand = _med_rec()
        attempt_medication_record_write(cand, empty_store, ddi_mode="force_high")
        events = empty_store.read_ledger_events()
        for e in events:
            safe_str = str(e.get("safe_public_details", ""))
            assert "@" not in safe_str
            assert "DOB" not in safe_str.upper()
            assert "MRN" not in safe_str.upper()


# ---------------------------------------------------------------------------
# Ledger event helper tests
# ---------------------------------------------------------------------------


class TestLedgerHelpers:
    def test_ddi_block_event_type(self):
        from clinical_knowledge.ledger import make_ddi_block_event
        evt = make_ddi_block_event(
            record_id="rid1",
            safe_record_id="safe_rid1",
            severity="high",
            action="block_requires_confirmation",
            safe_ddi_summary={"synthetic": True},
            explanation="test",
        )
        assert evt.event_type == LedgerEventType.DDI_BLOCK

    def test_ddi_warning_event_type(self):
        from clinical_knowledge.ledger import make_ddi_warning_event
        evt = make_ddi_warning_event(
            record_id="rid1",
            safe_record_id="safe_rid1",
            severity="medium",
            action="warn_requires_ack",
            safe_ddi_summary={"synthetic": True},
            explanation="test",
        )
        assert evt.event_type == LedgerEventType.DDI_WARNING

    def test_ddi_block_safe_public_details_no_raw_data(self):
        from clinical_knowledge.ledger import make_ddi_block_event
        evt = make_ddi_block_event(
            record_id="rid1",
            safe_record_id="safe_rid1",
            severity="high",
            action="block_requires_confirmation",
            safe_ddi_summary={"severity": "high", "synthetic": True},
            explanation="HIGH severity test",
        )
        details_str = str(evt.safe_public_details)
        assert "source_ref" not in details_str
        assert "patient" not in details_str.lower()
        assert evt.safe_public_details["severity"] == "high"


# ---------------------------------------------------------------------------
# Safety boundary tests
# ---------------------------------------------------------------------------


class TestSafetyBoundaries:
    def test_no_medication_recommendation_generated(self, empty_store):
        cand = _med_rec("synth_med_alpha")
        result = attempt_medication_record_write(
            cand, empty_store, ddi_mode="force_high"
        )
        expl = result.explanation.lower()
        assert "you should" not in expl
        assert "recommend" not in expl
        assert "prescribe" not in expl
        assert "take" not in expl

    def test_no_prescription_dosing_advice(self, empty_store):
        cand = _med_rec()
        result = attempt_medication_record_write(
            cand, empty_store, ddi_mode="force_medium"
        )
        expl = result.explanation.lower()
        assert "mg" not in expl
        assert "dose" not in expl

    def test_high_never_auto_accepted(self, empty_store):
        cand = _med_rec()
        result = attempt_medication_record_write(
            cand, empty_store, ddi_mode="force_high", user_confirmed_high=False
        )
        assert result.allowed_to_write is False

    def test_medium_never_auto_accepted(self, empty_store):
        cand = _med_rec()
        result = attempt_medication_record_write(
            cand, empty_store, ddi_mode="force_medium", user_acknowledged=False
        )
        assert result.allowed_to_write is False

    def test_unavailable_not_silently_accepted(self, empty_store):
        cand = _med_rec()
        result = attempt_medication_record_write(
            cand, empty_store, ddi_mode="unavailable"
        )
        assert result.allowed_to_write is False
        active_ids = {r["record_id"] for r in empty_store.list_active()}
        assert cand.record_id not in active_ids


# ---------------------------------------------------------------------------
# Truth Resolution boundary tests
# ---------------------------------------------------------------------------


class TestTruthResolutionBoundary:
    def test_dose_conflict_still_quarantines(self):
        from datetime import datetime, timezone
        from clinical_knowledge.truth_resolution.engine import apply_truth_resolution
        from clinical_knowledge.truth_resolution.models import ResolutionAction, ResolutionRule

        store = MKBStore(db_path=":memory:")
        ts = datetime.now(timezone.utc).isoformat()

        exist = MKBRecord(
            record_id=new_record_id(),
            safe_record_id=make_safe_record_id(new_record_id()),
            session_id="b05_tr",
            fact_type="medication_antiepileptic",
            entity_text="synth_med_alpha 500mg twice daily",
            trust_level=TrustLevel.OPERATOR_REVIEWED,
            tier=KnowledgeTier.ACTIVE,
            status=RecordStatus.CONFIRMED,
            source_type=SourceType.SYNTHETIC,
            confidence=0.80,
            created_at=ts,
        )
        cand = MKBRecord(
            record_id=new_record_id(),
            safe_record_id=make_safe_record_id(new_record_id()),
            session_id="b05_tr",
            fact_type="medication_antiepileptic",
            entity_text="synth_med_alpha 1000mg once daily",
            trust_level=TrustLevel.OPERATOR_REVIEWED,
            tier=KnowledgeTier.ACTIVE,
            status=RecordStatus.CONFIRMED,
            source_type=SourceType.SYNTHETIC,
            confidence=0.80,
            created_at=ts,
        )
        store.insert_record(exist)
        result = apply_truth_resolution(cand, exist, store)
        assert result is not None
        assert result.resolution == ResolutionAction.QUARANTINE
        assert result.rule_applied == ResolutionRule.MEDICATION_DOSE_CONFLICT

    def test_truth_resolution_does_not_invoke_ddi(self):
        from datetime import datetime, timezone
        from clinical_knowledge.truth_resolution.engine import apply_truth_resolution

        store = MKBStore(db_path=":memory:")
        ts = datetime.now(timezone.utc).isoformat()

        exist = MKBRecord(
            record_id=new_record_id(),
            safe_record_id=make_safe_record_id(new_record_id()),
            session_id="b05_ddi",
            fact_type="medication_antiepileptic",
            entity_text="synth_med_alpha 500mg twice daily",
            trust_level=TrustLevel.OPERATOR_REVIEWED,
            tier=KnowledgeTier.ACTIVE,
            status=RecordStatus.CONFIRMED,
            source_type=SourceType.SYNTHETIC,
            confidence=0.80,
            created_at=ts,
        )
        cand = MKBRecord(
            record_id=new_record_id(),
            safe_record_id=make_safe_record_id(new_record_id()),
            session_id="b05_ddi",
            fact_type="medication_antiepileptic",
            entity_text="synth_med_alpha 1000mg once daily",
            trust_level=TrustLevel.OPERATOR_REVIEWED,
            tier=KnowledgeTier.ACTIVE,
            status=RecordStatus.CONFIRMED,
            source_type=SourceType.SYNTHETIC,
            confidence=0.80,
            created_at=ts,
        )
        store.insert_record(exist)
        apply_truth_resolution(cand, exist, store)

        events = store.read_ledger_events()
        event_types = [e["event_type"] for e in events]
        assert LedgerEventType.DDI_BLOCK.value not in event_types
        assert LedgerEventType.DDI_WARNING.value not in event_types


# ---------------------------------------------------------------------------
# Validation script smoke test
# ---------------------------------------------------------------------------


class TestValidationScript:
    def test_validation_script_succeeds(self):
        import importlib.util
        import tempfile
        from pathlib import Path

        script = Path(__file__).parent.parent / "scripts" / "run_cka_block05_medication_safety_validation.py"
        spec = importlib.util.spec_from_file_location("val_b05", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmpdir:
            report = mod.run_validation(report_dir=Path(tmpdir))
        assert report["all_cases_passed"] is True
        assert report["conclusion"] == "cka_b05_medication_safety_ready"

    def test_final_report_no_private_strings(self):
        import importlib.util
        import tempfile
        from pathlib import Path

        script = Path(__file__).parent.parent / "scripts" / "run_cka_block05_medication_safety_validation.py"
        spec = importlib.util.spec_from_file_location("val_b05_priv", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmpdir:
            report = mod.run_validation(report_dir=Path(tmpdir))

        from clinical_knowledge.privacy.report_privacy import check_public_report_payload
        check = check_public_report_payload(report)
        assert check.passed
        assert not check.raw_phi_logged_in_public_reports
        assert check.private_filename_path_leaks == 0
