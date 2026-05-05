"""Quarantine helpers for CKA-B04."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from clinical_knowledge.models import KnowledgeTier, RecordStatus
from clinical_knowledge.truth_resolution.models import TruthResolutionResult

if TYPE_CHECKING:
    from clinical_knowledge.store import MKBStore


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def apply_quarantine(
    result: TruthResolutionResult,
    store: "MKBStore",
) -> None:
    """Quarantine all records listed in result.quarantined_record_ids."""
    now = _now_utc()
    for rid in result.quarantined_record_ids:
        store.update_record_tier(rid, KnowledgeTier.QUARANTINED, now)
        store.update_record_status(rid, RecordStatus.ARCHIVED, now)


def apply_supersede(
    result: TruthResolutionResult,
    store: "MKBStore",
) -> None:
    """Supersede all records listed in result.superseded_record_ids."""
    now = _now_utc()
    for rid in result.superseded_record_ids:
        store.update_record_tier(rid, KnowledgeTier.SUPERSEDED, now)
        store.update_record_status(rid, RecordStatus.ARCHIVED, now)
