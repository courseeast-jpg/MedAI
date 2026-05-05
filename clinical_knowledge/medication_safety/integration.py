"""Store integration helpers for CKA-B05 Medication Safety Gate."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, List, Optional

from clinical_knowledge.ledger import make_ddi_block_event, make_ddi_warning_event
from clinical_knowledge.medication_safety.models import (
    DDICheckStatus,
    DDISeverity,
    MedicationSafetyAction,
    MedicationWriteGateResult,
)
from clinical_knowledge.medication_safety.write_gate import (
    evaluate_medication_write_gate,
    _is_medication_fact,
)
from clinical_knowledge.models import KnowledgeTier, RecordStatus

if TYPE_CHECKING:
    from clinical_knowledge.store import MKBStore


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def attempt_medication_record_write(
    candidate_fact,
    store: "MKBStore",
    *,
    ddi_mode: str = "normal",
    active_medications: Optional[List[str]] = None,
    user_acknowledged: bool = False,
    user_confirmed_high: bool = False,
) -> MedicationWriteGateResult:
    """Attempt to write a medication fact through the DDI gate.

    Non-medication facts bypass the gate and are inserted directly.
    Returns MedicationWriteGateResult describing outcome.
    """
    if active_medications is None:
        active_medications = []

    # Non-medication facts: bypass DDI gate, insert and return clear result
    if not _is_medication_fact(candidate_fact):
        store.insert_record(candidate_fact)
        return MedicationWriteGateResult(
            action=MedicationSafetyAction.ALLOW,
            allowed_to_write=True,
            requires_user_confirmation=False,
            candidate_status="active",
            ddi_checked=False,
            ddi_status=DDICheckStatus.CLEAR,
            ddi_findings=[],
            ledger_event_ready=False,
            explanation="Non-medication fact — DDI gate bypassed.",
        )

    gate = evaluate_medication_write_gate(
        candidate_fact, active_medications, ddi_mode=ddi_mode
    )

    record_id = getattr(candidate_fact, "record_id", "") or ""
    safe_id = getattr(candidate_fact, "safe_record_id", "") or ""
    now = _now_utc()

    action = gate.action

    # HIGH — block unless user explicitly confirmed
    if action == MedicationSafetyAction.BLOCK_REQUIRES_CONFIRMATION:
        evt = make_ddi_block_event(
            record_id=record_id,
            safe_record_id=safe_id,
            severity=DDISeverity.HIGH.value,
            action=action.value,
            safe_ddi_summary=gate.safe_public_summary,
            explanation=gate.explanation,
        )
        store.append_ledger_event(evt)

        if user_confirmed_high:
            # Write with requires_review=True — never auto-accept
            _force_requires_review(candidate_fact)
            store.insert_record(candidate_fact)
            return MedicationWriteGateResult(
                action=action,
                allowed_to_write=True,
                requires_user_confirmation=True,
                candidate_status="high_blocked",
                ddi_checked=True,
                ddi_status=DDICheckStatus.HIGH_BLOCKED,
                ddi_findings=gate.ddi_findings,
                ledger_event_ready=True,
                explanation=gate.explanation,
                safe_public_summary=gate.safe_public_summary,
            )

        return gate  # not allowed

    # MEDIUM — warn unless user acknowledged
    if action == MedicationSafetyAction.WARN_REQUIRES_ACK:
        evt = make_ddi_warning_event(
            record_id=record_id,
            safe_record_id=safe_id,
            severity=DDISeverity.MEDIUM.value,
            action=action.value,
            safe_ddi_summary=gate.safe_public_summary,
            explanation=gate.explanation,
        )
        store.append_ledger_event(evt)

        if user_acknowledged:
            _force_requires_review(candidate_fact)
            store.insert_record(candidate_fact)
            return MedicationWriteGateResult(
                action=action,
                allowed_to_write=True,
                requires_user_confirmation=True,
                candidate_status="pending_ddi_ack",
                ddi_checked=True,
                ddi_status=DDICheckStatus.MEDIUM,
                ddi_findings=gate.ddi_findings,
                ledger_event_ready=True,
                explanation=gate.explanation,
                safe_public_summary=gate.safe_public_summary,
            )

        return gate  # not allowed

    # LOW — allow with note, insert directly
    if action == MedicationSafetyAction.ALLOW_WITH_NOTE:
        store.insert_record(candidate_fact)
        return gate

    # NONE (ALLOW) — insert directly
    if action == MedicationSafetyAction.ALLOW:
        store.insert_record(candidate_fact)
        return gate

    # UNAVAILABLE — queue, do not insert active record; log warning event
    evt = make_ddi_warning_event(
        record_id=record_id,
        safe_record_id=safe_id,
        severity=DDISeverity.UNAVAILABLE.value,
        action=action.value,
        safe_ddi_summary=gate.safe_public_summary,
        explanation=gate.explanation,
    )
    store.append_ledger_event(evt)
    return gate  # not written to active store


def _force_requires_review(record) -> None:
    """Force requires_review=True on a record before writing."""
    try:
        object.__setattr__(record, "requires_review", True)
    except (AttributeError, TypeError):
        try:
            record.requires_review = True
        except AttributeError:
            pass
