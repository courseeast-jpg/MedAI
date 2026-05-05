"""Integration helpers for CKA-B04 Truth Resolution.

Lightweight helpers for future Quality Gate / Controlled Enrichment paths
and minimal Decision Engine integration.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from clinical_knowledge.truth_resolution.models import (
    ResolutionAction,
    TruthResolutionResult,
)

if TYPE_CHECKING:
    from clinical_knowledge.store import MKBStore


def check_candidate_before_insert(candidate, store: "MKBStore") -> Optional[dict]:
    """Check if a candidate conflicts with any existing active record.

    Returns a safe escalation summary if a conflict is found.
    Returns None if no conflict — safe to insert.
    Does NOT insert the record.
    """
    from clinical_knowledge.truth_resolution.conflict_detection import detect_conflict
    from clinical_knowledge.truth_resolution.engine import resolve_conflict

    for existing_dict in store.list_active():
        from clinical_knowledge.models import (
            DDIStatus, KnowledgeTier, MKBRecord,
            RecordStatus, SourceType, TrustLevel,
        )
        try:
            existing = MKBRecord(
                record_id=existing_dict.get("record_id", ""),
                safe_record_id=existing_dict.get("safe_record_id", ""),
                session_id=existing_dict.get("session_id", ""),
                fact_type=existing_dict.get("fact_type", ""),
                entity_text=existing_dict.get("entity_text", ""),
                structured=existing_dict.get("structured") or {},
                specialty=existing_dict.get("specialty", "general"),
                source_type=SourceType(existing_dict.get("source_type", "unknown")),
                trust_level=TrustLevel(existing_dict.get("trust_level", 5)),
                tier=KnowledgeTier(existing_dict.get("tier", "hypothesis")),
                status=RecordStatus(existing_dict.get("status", "pending")),
                confidence=existing_dict.get("confidence", 0.0),
                created_at=existing_dict.get("created_at", ""),
                updated_at=existing_dict.get("updated_at", ""),
            )
        except Exception:
            continue

        pair = detect_conflict(candidate, existing)
        if pair is not None:
            result = resolve_conflict(pair)
            return _safe_escalation_summary(result)

    return None


def _safe_escalation_summary(result: TruthResolutionResult) -> dict:
    return {
        "conflict_detected": True,
        "rule_applied": result.rule_applied.value,
        "resolution": result.resolution.value,
        "requires_review": result.requires_review,
        "confidence": result.confidence,
        "safe_public_summary": result.safe_public_summary,
    }


def active_context_is_clean(store: "MKBStore") -> bool:
    """Verify active context excludes quarantined and superseded records."""
    active = store.list_active()
    quarantined = store.list_quarantined()
    superseded = store.list_superseded()
    active_ids = {r.get("record_id") for r in active}
    quarantined_ids = {r.get("record_id") for r in quarantined}
    superseded_ids = {r.get("record_id") for r in superseded}
    overlap_q = active_ids & quarantined_ids
    overlap_s = active_ids & superseded_ids
    return len(overlap_q) == 0 and len(overlap_s) == 0
