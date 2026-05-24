"""Focused tests for app/operator_review_actions.py.

MEDAI-CORPUS-EXTRACTION-TO-MKB-MINIMUM-03-OPERATOR-REVIEW-UX.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import TIER_ACTIVE, TIER_QUARANTINED, TIER_SUPERSEDED, TRUST_CLINICAL
from app.operator_review_actions import (
    ACCEPT_DISCLAIMER,
    BLOCKED_DDI_STATES,
    DEFER_DISCLAIMER,
    LEDGER_EVENT_TYPE,
    LEDGER_SOURCE,
    REJECT_DISCLAIMER,
    accept_after_source_comparison,
    defer_extracted_fact,
    reject_extracted_fact,
    render_action_affordances_plan,
)
from app.schemas import MKBRecord
from mkb.sqlite_store import SQLiteStore


@pytest.fixture()
def store(tmp_path) -> SQLiteStore:
    return SQLiteStore(db_path=tmp_path / "mkb.db", encryption_key="")


def _make_review_bound_lab(store, test_name="Glucose", value="5.4", unit="mmol/L"):
    record = MKBRecord(
        fact_type="test_result",
        content=f"Test: {test_name}: {value} {unit}",
        structured={
            "text": test_name,
            "test_name": test_name,
            "value": value,
            "unit": unit,
            "parser_name": "deterministic_lab_line_adapter",
            "requires_human_review": True,
            "auto_accept_allowed": False,
        },
        specialty="general",
        source_type="extraction",
        source_name="synthetic-test-001",
        trust_level=TRUST_CLINICAL,
        confidence=0.55,
        tier=TIER_QUARANTINED,
        extraction_method="rules_based",
        requires_review=True,
        ddi_checked=False,
    )
    store.write_record(record, session_id="syn")
    return record


def _ledger_event_count(store) -> int:
    with store._get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM ledger WHERE event_type=?",
            (LEDGER_EVENT_TYPE,),
        ).fetchone()
    return int(row["c"] if isinstance(row, dict) else row[0])


def test_accept_after_source_comparison_moves_to_active(store):
    rec = _make_review_bound_lab(store)
    result = accept_after_source_comparison(store, rec.id, operator_note="ok", session_id="s")
    assert result.success is True
    assert result.previous_tier == TIER_QUARANTINED
    assert result.previous_requires_review is True
    assert result.new_tier == TIER_ACTIVE
    assert result.new_status == "accepted_after_operator_review"
    assert result.new_requires_review is False
    assert result.operator_note_present is True
    assert result.error_code is None
    assert result.ledger_event_id is not None
    assert result.safe_message == ACCEPT_DISCLAIMER

    persisted = store.get_record(rec.id)
    assert persisted is not None
    assert persisted.tier == TIER_ACTIVE
    assert persisted.status == "accepted_after_operator_review"
    assert persisted.requires_review is False
    assert persisted.structured["operator_review_status"] == "accepted_after_operator_review"
    assert persisted.structured["operator_action"] == "accept_after_source_comparison"
    assert persisted.structured["operator_note_present"] is True
    # Fact content / value / unit are preserved verbatim.
    assert persisted.structured["test_name"] == "Glucose"
    assert persisted.structured["value"] == "5.4"
    assert persisted.structured["unit"] == "mmol/L"


def test_reject_extracted_fact_marks_record_rejected_and_not_active(store):
    rec = _make_review_bound_lab(store, test_name="WBC", value="7.2", unit="x10E9/L")
    result = reject_extracted_fact(store, rec.id, session_id="s")
    assert result.success is True
    assert result.new_tier == TIER_SUPERSEDED
    assert result.new_status == "rejected_after_operator_review"
    assert result.new_requires_review is False
    assert result.safe_message == REJECT_DISCLAIMER

    persisted = store.get_record(rec.id)
    assert persisted is not None
    assert persisted.tier == TIER_SUPERSEDED
    assert persisted.status == "rejected_after_operator_review"
    assert persisted.requires_review is False
    # Rejected records do not appear in get_records_requiring_review.
    assert all(r.id != rec.id for r in store.get_records_requiring_review())


def test_defer_extracted_fact_keeps_record_review_bound(store):
    rec = _make_review_bound_lab(store, test_name="Hemoglobin", value="13.5", unit="g/dL")
    result = defer_extracted_fact(store, rec.id, operator_note="defer", session_id="s")
    assert result.success is True
    assert result.new_tier == TIER_QUARANTINED
    assert result.new_status == "deferred_by_operator"
    assert result.new_requires_review is True
    assert result.safe_message == DEFER_DISCLAIMER

    persisted = store.get_record(rec.id)
    assert persisted is not None
    assert persisted.tier == TIER_QUARANTINED
    assert persisted.requires_review is True
    assert persisted.status == "deferred_by_operator"
    assert any(r.id == rec.id for r in store.get_records_requiring_review())


def test_medication_fact_cannot_be_accepted_through_lab_review(store):
    rec = MKBRecord(
        fact_type="medication",
        content="Medication: ibuprofen 200mg",
        structured={"name": "ibuprofen", "dose": "200mg"},
        specialty="general",
        source_type="extraction",
        source_name="synthetic-med-001",
        trust_level=TRUST_CLINICAL,
        confidence=0.7,
        tier=TIER_QUARANTINED,
        extraction_method="rules_based",
        requires_review=True,
        ddi_checked=False,
    )
    store.write_record(rec, session_id="syn")
    result = accept_after_source_comparison(store, rec.id)
    assert result.success is False
    assert result.error_code == "medication_safety_workflow_required"
    persisted = store.get_record(rec.id)
    assert persisted.tier == TIER_QUARANTINED
    assert persisted.requires_review is True


def test_ddi_blocked_or_pending_medication_record_cannot_be_accepted(store):
    rec = MKBRecord(
        fact_type="medication",
        content="Medication: warfarin",
        structured={"name": "warfarin"},
        specialty="general",
        source_type="extraction",
        source_name="syn-med-002",
        trust_level=TRUST_CLINICAL,
        confidence=0.7,
        tier=TIER_QUARANTINED,
        extraction_method="rules_based",
        requires_review=True,
        ddi_checked=False,
        ddi_status="high_blocked",
    )
    store.write_record(rec, session_id="syn")
    result = accept_after_source_comparison(store, rec.id)
    assert result.success is False
    # Medication guard fires before DDI guard, but either is acceptable.
    assert result.error_code in {"medication_safety_workflow_required", "ddi_blocked_or_pending"}


def test_ddi_blocked_state_blocks_accept_for_test_result_too(store):
    rec = _make_review_bound_lab(store)
    # Mutate persisted record to look DDI-blocked (synthetic edge case).
    rec.ddi_status = "high_blocked"
    store.write_record(rec, session_id="s")
    result = accept_after_source_comparison(store, rec.id)
    assert result.success is False
    assert result.error_code == "ddi_blocked_or_pending"
    persisted = store.get_record(rec.id)
    assert persisted.tier == TIER_QUARANTINED


def test_missing_record_fails_safely(store):
    result = accept_after_source_comparison(store, "no-such-id")
    assert result.success is False
    assert result.error_code == "record_not_found"
    assert result.previous_tier is None
    assert result.new_tier is None


def test_already_active_record_cannot_be_accepted_again(store):
    rec = _make_review_bound_lab(store)
    first = accept_after_source_comparison(store, rec.id)
    assert first.success is True
    second = accept_after_source_comparison(store, rec.id)
    assert second.success is False
    assert second.error_code == "record_already_active"


def test_action_result_does_not_contain_raw_source_text(store):
    rec = _make_review_bound_lab(store)
    operator_note = "Compared with source PDF line 17"
    result = accept_after_source_comparison(store, rec.id, operator_note=operator_note)
    serialized = repr(result.to_public_dict())
    # The note text itself must not appear in the public dict.
    assert operator_note not in serialized
    # Only the presence flag is exposed.
    assert "operator_note_present" in serialized
    assert "Compared" not in serialized


def test_render_action_affordances_plan_marks_actions_for_review_bound_test_result():
    plan = render_action_affordances_plan(
        {
            "record_id": "r1",
            "fact_type": "test_result",
            "tier": TIER_QUARANTINED,
            "status": "active",
            "requires_review": True,
        }
    )
    assert plan["record_id"] == "r1"
    assert plan["auto_accept_allowed"] is False
    assert plan["review_required"] is True
    actions = {a["key"]: a for a in plan["actions"]}
    assert actions["accept_after_source_comparison"]["enabled"] is True
    assert actions["reject_extracted_fact"]["enabled"] is True
    assert actions["defer_extracted_fact"]["enabled"] is True


def test_render_action_affordances_plan_disables_accept_for_active_record():
    plan = render_action_affordances_plan(
        {
            "record_id": "r2",
            "fact_type": "test_result",
            "tier": TIER_ACTIVE,
            "status": "accepted_after_operator_review",
            "requires_review": False,
        }
    )
    accept = next(a for a in plan["actions"] if a["key"] == "accept_after_source_comparison")
    assert accept["enabled"] is False
    assert accept["reason"] == "record_already_active"


def test_render_action_affordances_plan_disables_accept_for_medication():
    plan = render_action_affordances_plan(
        {
            "record_id": "r3",
            "fact_type": "medication",
            "tier": TIER_QUARANTINED,
            "status": "pending_medication_review",
            "requires_review": True,
        }
    )
    accept = next(a for a in plan["actions"] if a["key"] == "accept_after_source_comparison")
    reject = next(a for a in plan["actions"] if a["key"] == "reject_extracted_fact")
    defer = next(a for a in plan["actions"] if a["key"] == "defer_extracted_fact")
    assert accept["enabled"] is False
    assert accept["reason"] == "medication_safety_workflow_required"
    # Reject and defer must also refuse to operate on medication facts via
    # the lab review workflow.
    assert reject["enabled"] is False
    assert defer["enabled"] is False


def test_ledger_event_count_increments_per_action(store):
    rec_accept = _make_review_bound_lab(store, "Glucose", "5.4", "mmol/L")
    rec_reject = _make_review_bound_lab(store, "WBC", "7.2", "x10E9/L")
    rec_defer = _make_review_bound_lab(store, "Hemoglobin", "13.5", "g/dL")
    assert _ledger_event_count(store) == 0
    accept_after_source_comparison(store, rec_accept.id, operator_note="ok")
    assert _ledger_event_count(store) == 1
    reject_extracted_fact(store, rec_reject.id)
    assert _ledger_event_count(store) == 2
    defer_extracted_fact(store, rec_defer.id)
    assert _ledger_event_count(store) == 3


def test_ledger_event_payload_contains_no_operator_note_text(store):
    rec = _make_review_bound_lab(store)
    operator_note = "Confirmed against page 3 of source document"
    accept_after_source_comparison(store, rec.id, operator_note=operator_note)
    with store._get_conn() as conn:
        row = conn.execute(
            "SELECT details_json, previous_value FROM ledger WHERE event_type=?",
            (LEDGER_EVENT_TYPE,),
        ).fetchone()
    payload = row["details_json"] if isinstance(row, dict) else row[0]
    previous = row["previous_value"] if isinstance(row, dict) else row[1]
    combined = f"{payload}|{previous}"
    assert operator_note not in combined
    # Only the presence flag and length bucket are written.
    assert "operator_note_present" in combined
    assert "operator_note_length_bucket" in combined


def test_blocked_ddi_states_constant_is_defensive():
    assert "high_blocked" in BLOCKED_DDI_STATES
    assert "pending_ddi" in BLOCKED_DDI_STATES
    assert "pending_ddi_check" in BLOCKED_DDI_STATES
    assert "pending_medication_review" in BLOCKED_DDI_STATES


def test_ledger_source_marker_is_operator_review_ux():
    assert LEDGER_SOURCE == "operator_review_ux"
