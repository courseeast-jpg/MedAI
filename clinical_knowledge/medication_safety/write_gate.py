"""Layer 2 Medication Safety Write Gate for CKA-B05.

Evaluates whether a medication fact may be written to the active MKB.
Does NOT bypass Truth Resolution.
Does NOT provide prescribing advice or medication recommendations.
"""
from __future__ import annotations

from typing import List

from clinical_knowledge.medication_safety.ddi_stub import check_ddi_stub
from clinical_knowledge.medication_safety.models import (
    DDICheckStatus,
    DDISeverity,
    MedicationSafetyAction,
    MedicationWriteGateResult,
)

_MEDICATION_FACT_TYPES = frozenset({
    "medication", "drug", "prescription", "antiepileptic", "anticonvulsant",
    "pharmaceutical", "medication_antiepileptic", "medication_reference",
})


def _is_medication_fact(candidate_fact) -> bool:
    ft = (getattr(candidate_fact, "fact_type", "") or "").lower().strip()
    return ft in _MEDICATION_FACT_TYPES or ft.startswith("medication")


def evaluate_medication_write_gate(
    candidate_fact,
    active_medications: List[str],
    *,
    ddi_mode: str = "normal",
) -> MedicationWriteGateResult:
    """Evaluate whether candidate medication fact may be written.

    Trigger: candidate_fact.fact_type == "medication" (or medication-typed facts).
    Returns gate result without modifying the store.
    """
    entity = getattr(candidate_fact, "entity_text", "") or ""
    safe_id = getattr(candidate_fact, "safe_record_id", "") or ""

    ddi_result = check_ddi_stub(entity, active_medications, mode=ddi_mode)
    severity = ddi_result.highest_severity

    if severity == DDISeverity.HIGH:
        return MedicationWriteGateResult(
            action=MedicationSafetyAction.BLOCK_REQUIRES_CONFIRMATION,
            allowed_to_write=False,
            requires_user_confirmation=True,
            candidate_status="blocked_ddi",
            ddi_checked=True,
            ddi_status=DDICheckStatus.HIGH_BLOCKED,
            ddi_findings=ddi_result.findings,
            ledger_event_ready=True,
            explanation=(
                "HIGH severity synthetic interaction requires explicit user "
                "confirmation before medication fact can be written."
            ),
            safe_public_summary=_safe_summary(safe_id, severity, MedicationSafetyAction.BLOCK_REQUIRES_CONFIRMATION, ddi_result),
        )

    if severity == DDISeverity.MEDIUM:
        return MedicationWriteGateResult(
            action=MedicationSafetyAction.WARN_REQUIRES_ACK,
            allowed_to_write=False,
            requires_user_confirmation=True,
            candidate_status="pending_ddi_ack",
            ddi_checked=True,
            ddi_status=DDICheckStatus.MEDIUM,
            ddi_findings=ddi_result.findings,
            ledger_event_ready=True,
            explanation=(
                "MEDIUM severity synthetic interaction requires user "
                "acknowledgment before write."
            ),
            safe_public_summary=_safe_summary(safe_id, severity, MedicationSafetyAction.WARN_REQUIRES_ACK, ddi_result),
        )

    if severity == DDISeverity.LOW:
        return MedicationWriteGateResult(
            action=MedicationSafetyAction.ALLOW_WITH_NOTE,
            allowed_to_write=True,
            requires_user_confirmation=False,
            candidate_status="active",
            ddi_checked=True,
            ddi_status=DDICheckStatus.LOW,
            ddi_findings=ddi_result.findings,
            ledger_event_ready=False,
            explanation="LOW severity synthetic interaction noted; write allowed.",
            safe_public_summary=_safe_summary(safe_id, severity, MedicationSafetyAction.ALLOW_WITH_NOTE, ddi_result),
        )

    if severity == DDISeverity.NONE:
        return MedicationWriteGateResult(
            action=MedicationSafetyAction.ALLOW,
            allowed_to_write=True,
            requires_user_confirmation=False,
            candidate_status="active",
            ddi_checked=True,
            ddi_status=DDICheckStatus.CLEAR,
            ddi_findings=[],
            ledger_event_ready=False,
            explanation="No synthetic DDI interaction detected; write allowed.",
            safe_public_summary=_safe_summary(safe_id, severity, MedicationSafetyAction.ALLOW, ddi_result),
        )

    # UNAVAILABLE
    return MedicationWriteGateResult(
        action=MedicationSafetyAction.QUEUE_PENDING_DDI,
        allowed_to_write=False,
        requires_user_confirmation=False,
        candidate_status="pending_ddi_check",
        ddi_checked=False,
        ddi_status=DDICheckStatus.PENDING,
        ddi_findings=[],
        ledger_event_ready=True,
        explanation=(
            "DDI check unavailable. Medication write queued pending safety check."
        ),
        safe_public_summary=_safe_summary(safe_id, severity, MedicationSafetyAction.QUEUE_PENDING_DDI, ddi_result),
    )


def _safe_summary(
    safe_id: str,
    severity: DDISeverity,
    action: MedicationSafetyAction,
    ddi_result,
) -> dict:
    return {
        "candidate_safe_id": safe_id,
        "severity": severity.value,
        "action": action.value,
        "synthetic": True,
        "finding_count": len(ddi_result.findings),
    }
