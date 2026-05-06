"""Immutable ledger for CKA — every write/update appends a LedgerEvent.

The ledger is append-only. Events are written to the SQLite store via the
store interface. This module provides the event-creation helpers.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from clinical_knowledge.models import LedgerEvent, LedgerEventType
from clinical_knowledge.safe_ids import new_event_id


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_created_event(
    record_id: str,
    safe_record_id: str,
    tier: str,
    trust_level: int,
    actor: str = "system",
    reason: str = "record created",
) -> LedgerEvent:
    return LedgerEvent(
        event_id=new_event_id(),
        event_type=LedgerEventType.MKB_RECORD_CREATED,
        record_id=record_id,
        timestamp=_now_utc(),
        actor=actor,
        reason=reason,
        details={"tier": tier, "trust_level": trust_level},
        safe_public_details={"safe_record_id": safe_record_id, "tier": tier},
    )


def make_updated_event(
    record_id: str,
    safe_record_id: str,
    changed_fields: list[str],
    actor: str = "system",
    reason: str = "record updated",
) -> LedgerEvent:
    return LedgerEvent(
        event_id=new_event_id(),
        event_type=LedgerEventType.MKB_RECORD_UPDATED,
        record_id=record_id,
        timestamp=_now_utc(),
        actor=actor,
        reason=reason,
        details={"changed_fields": changed_fields},
        safe_public_details={"safe_record_id": safe_record_id, "changed_fields": changed_fields},
    )


def make_tier_changed_event(
    record_id: str,
    safe_record_id: str,
    old_tier: str,
    new_tier: str,
    actor: str = "system",
    reason: str = "tier changed",
) -> LedgerEvent:
    return LedgerEvent(
        event_id=new_event_id(),
        event_type=LedgerEventType.TIER_CHANGED,
        record_id=record_id,
        timestamp=_now_utc(),
        actor=actor,
        reason=reason,
        details={"old_tier": old_tier, "new_tier": new_tier},
        safe_public_details={"safe_record_id": safe_record_id, "old_tier": old_tier, "new_tier": new_tier},
    )


def make_status_changed_event(
    record_id: str,
    safe_record_id: str,
    old_status: str,
    new_status: str,
    actor: str = "system",
    reason: str = "status changed",
) -> LedgerEvent:
    return LedgerEvent(
        event_id=new_event_id(),
        event_type=LedgerEventType.STATUS_CHANGED,
        record_id=record_id,
        timestamp=_now_utc(),
        actor=actor,
        reason=reason,
        details={"old_status": old_status, "new_status": new_status},
        safe_public_details={"safe_record_id": safe_record_id, "old_status": old_status, "new_status": new_status},
    )


def make_privacy_audit_event(
    record_id: str,
    safe_record_id: str,
    findings_summary: Dict[str, Any],
    passed: bool,
    actor: str = "privacy_boundary",
) -> LedgerEvent:
    return LedgerEvent(
        event_id=new_event_id(),
        event_type=LedgerEventType.PRIVACY_AUDIT,
        record_id=record_id,
        timestamp=_now_utc(),
        actor=actor,
        reason="privacy audit",
        details={"findings_summary": findings_summary, "passed": passed},
        safe_public_details={
            "safe_record_id": safe_record_id,
            "passed": passed,
            "finding_categories": list(findings_summary.keys()),
        },
    )


def make_safe_mode_entry_event(
    record_id: str,
    safe_record_id: str,
    reason: str,
    actor: str = "decision_engine",
) -> LedgerEvent:
    return LedgerEvent(
        event_id=new_event_id(),
        event_type=LedgerEventType.SAFE_MODE_ENTRY,
        record_id=record_id,
        timestamp=_now_utc(),
        actor=actor,
        reason=reason,
        details={"safe_mode_reason": reason},
        safe_public_details={"safe_record_id": safe_record_id, "safe_mode_reason": reason},
    )


def make_response_discarded_event(
    record_id: str,
    safe_record_id: str,
    score: float,
    reason: str,
    actor: str = "decision_engine",
) -> LedgerEvent:
    return LedgerEvent(
        event_id=new_event_id(),
        event_type=LedgerEventType.RESPONSE_DISCARDED,
        record_id=record_id,
        timestamp=_now_utc(),
        actor=actor,
        reason=reason,
        details={"score": score, "discard_reason": reason},
        safe_public_details={"safe_record_id": safe_record_id, "score": score},
    )


def make_truth_resolution_event(
    record_id: str,
    safe_record_id: str,
    rule_applied: str,
    resolution: str,
    winner_safe_id: str,
    loser_safe_id: str,
    confidence: float,
    requires_review: bool,
    explanation: str,
    actor: str = "truth_resolution_engine",
) -> LedgerEvent:
    return LedgerEvent(
        event_id=new_event_id(),
        event_type=LedgerEventType.TRUTH_RESOLUTION,
        record_id=record_id,
        timestamp=_now_utc(),
        actor=actor,
        reason=f"truth_resolution:{rule_applied}",
        details={
            "rule_applied": rule_applied,
            "resolution": resolution,
            "winner_safe_id": winner_safe_id,
            "loser_safe_id": loser_safe_id,
            "confidence": confidence,
            "requires_review": requires_review,
            "explanation": explanation,
        },
        safe_public_details={
            "safe_record_id": safe_record_id,
            "rule_applied": rule_applied,
            "resolution": resolution,
            "winner_safe_id": winner_safe_id,
            "loser_safe_id": loser_safe_id,
            "confidence": confidence,
            "requires_review": requires_review,
        },
    )


def make_quarantine_event(
    record_id: str,
    safe_record_id: str,
    quarantined_safe_ids: list,
    conflict_type: str,
    explanation: str,
    actor: str = "truth_resolution_engine",
) -> LedgerEvent:
    return LedgerEvent(
        event_id=new_event_id(),
        event_type=LedgerEventType.QUARANTINE,
        record_id=record_id,
        timestamp=_now_utc(),
        actor=actor,
        reason=f"quarantine:{conflict_type}",
        details={
            "quarantined_safe_ids": quarantined_safe_ids,
            "conflict_type": conflict_type,
            "explanation": explanation,
            "requires_review": True,
        },
        safe_public_details={
            "safe_record_id": safe_record_id,
            "quarantined_safe_ids": quarantined_safe_ids,
            "conflict_type": conflict_type,
            "requires_review": True,
        },
    )


def make_ddi_block_event(
    record_id: str,
    safe_record_id: str,
    severity: str,
    action: str,
    safe_ddi_summary: dict,
    explanation: str,
    actor: str = "medication_safety_gate",
) -> LedgerEvent:
    return LedgerEvent(
        event_id=new_event_id(),
        event_type=LedgerEventType.DDI_BLOCK,
        record_id=record_id,
        timestamp=_now_utc(),
        actor=actor,
        reason=f"ddi_block:severity={severity}",
        details={
            "severity": severity,
            "action": action,
            "explanation": explanation,
            "safe_ddi_summary": safe_ddi_summary,
        },
        safe_public_details={
            "safe_record_id": safe_record_id,
            "severity": severity,
            "action": action,
            "safe_ddi_summary": safe_ddi_summary,
        },
    )


def make_ddi_warning_event(
    record_id: str,
    safe_record_id: str,
    severity: str,
    action: str,
    safe_ddi_summary: dict,
    explanation: str,
    actor: str = "medication_safety_gate",
) -> LedgerEvent:
    return LedgerEvent(
        event_id=new_event_id(),
        event_type=LedgerEventType.DDI_WARNING,
        record_id=record_id,
        timestamp=_now_utc(),
        actor=actor,
        reason=f"ddi_warning:severity={severity}",
        details={
            "severity": severity,
            "action": action,
            "explanation": explanation,
            "safe_ddi_summary": safe_ddi_summary,
        },
        safe_public_details={
            "safe_record_id": safe_record_id,
            "severity": severity,
            "action": action,
            "safe_ddi_summary": safe_ddi_summary,
        },
    )


def make_medical_coding_event(
    record_id: str,
    safe_record_id: str,
    coding_status: str,
    systems_attempted: list,
    preferred_code_summary: Optional[Dict[str, Any]],
    actor: str = "medical_coding_service",
) -> LedgerEvent:
    return LedgerEvent(
        event_id=new_event_id(),
        event_type=LedgerEventType.MEDICAL_CODING,
        record_id=record_id,
        timestamp=_now_utc(),
        actor=actor,
        reason=f"medical_coding:{coding_status}",
        details={
            "coding_status": coding_status,
            "systems_attempted": systems_attempted,
            "preferred_code_summary": preferred_code_summary or {},
        },
        safe_public_details={
            "safe_record_id": safe_record_id,
            "coding_status": coding_status,
            "systems_attempted": systems_attempted,
            "preferred_code_summary": preferred_code_summary or {},
        },
    )


def make_enrichment_write_event(
    record_id: str,
    safe_record_id: str,
    tier: str,
    trust_level: int,
    safe_candidate_id: str,
    actor: str = "enrichment_writer",
) -> LedgerEvent:
    return LedgerEvent(
        event_id=new_event_id(),
        event_type=LedgerEventType.ENRICHMENT_WRITE,
        record_id=record_id,
        timestamp=_now_utc(),
        actor=actor,
        reason="enrichment hypothesis write",
        details={"tier": tier, "trust_level": trust_level, "safe_candidate_id": safe_candidate_id},
        safe_public_details={
            "safe_record_id": safe_record_id,
            "safe_candidate_id": safe_candidate_id,
            "tier": tier,
        },
    )


def make_hypothesis_promoted_event(
    record_id: str,
    safe_record_id: str,
    promotion_mode: str,
    actor: str = "enrichment_promotion",
) -> LedgerEvent:
    return LedgerEvent(
        event_id=new_event_id(),
        event_type=LedgerEventType.HYPOTHESIS_PROMOTED,
        record_id=record_id,
        timestamp=_now_utc(),
        actor=actor,
        reason=f"hypothesis_promotion:{promotion_mode}",
        details={"promotion_mode": promotion_mode},
        safe_public_details={
            "safe_record_id": safe_record_id,
            "promotion_mode": promotion_mode,
        },
    )


def make_validation_event(
    record_id: str,
    safe_record_id: str,
    validation_summary: Dict[str, Any],
    actor: str = "validation_script",
) -> LedgerEvent:
    return LedgerEvent(
        event_id=new_event_id(),
        event_type=LedgerEventType.VALIDATION_RUN,
        record_id=record_id,
        timestamp=_now_utc(),
        actor=actor,
        reason="validation run",
        details=validation_summary,
        safe_public_details={"safe_record_id": safe_record_id, **validation_summary},
    )
