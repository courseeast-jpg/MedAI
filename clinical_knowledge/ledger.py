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
