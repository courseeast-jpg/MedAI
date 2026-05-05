"""Enrichment integration orchestrator for CKA-B06.

Full pipeline:
  safe_mode_check → dedup_check → conflict_check → medication_safety_gate
  → hypothesis_write → ledger

Does NOT perform auto-promotion.
Does NOT write ACTIVE records.
Does NOT call real external APIs or LLMs.
"""
from __future__ import annotations

import re
from typing import List, Optional, TYPE_CHECKING

from clinical_knowledge.enrichment.enrichment_queue import EnrichmentQueue
from clinical_knowledge.enrichment.hypothesis_writer import write_candidate_as_hypothesis
from clinical_knowledge.enrichment.models import (
    EnrichmentAction,
    EnrichmentCandidate,
    EnrichmentCandidateStatus,
    EnrichmentWriteResult,
)

if TYPE_CHECKING:
    from clinical_knowledge.config import CKAConfig
    from clinical_knowledge.store import MKBStore


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().strip())


def _structured_core_value(structured: dict) -> str:
    """Return a simple string key for dedup based on structured content."""
    if not structured:
        return ""
    val = structured.get("value") or structured.get("source_count") or ""
    return str(val)


def is_duplicate_candidate(
    candidate: EnrichmentCandidate,
    existing_records: List[dict],
) -> bool:
    """Deterministic dedup: same specialty + fact_type + normalized entity + structured core.

    No vector similarity. No external dependencies.
    """
    c_spec = _normalize(candidate.specialty)
    c_ft = _normalize(candidate.fact_type)
    c_et = _normalize(candidate.entity_text)
    c_sv = _structured_core_value(candidate.structured)

    for rec in existing_records:
        r_spec = _normalize(rec.get("specialty", ""))
        r_ft = _normalize(rec.get("fact_type", ""))
        r_et = _normalize(rec.get("entity_text", ""))
        r_structured = rec.get("structured", {})
        if isinstance(r_structured, str):
            import json as _json
            try:
                r_structured = _json.loads(r_structured)
            except Exception:
                r_structured = {}
        r_sv = _structured_core_value(r_structured if isinstance(r_structured, dict) else {})

        if c_spec == r_spec and c_ft == r_ft and c_et == r_et and c_sv == r_sv:
            return True
    return False


def process_enrichment_candidate(
    candidate: EnrichmentCandidate,
    store: "MKBStore",
    queue: EnrichmentQueue,
    config: "CKAConfig",
    *,
    safe_mode: bool = False,
    ddi_mode: str = "normal",
    active_medications: Optional[List[str]] = None,
    user_acknowledged_medium: bool = False,
    user_confirmed_high: bool = False,
) -> EnrichmentWriteResult:
    """Process a single enrichment candidate through the full pipeline.

    Returns EnrichmentWriteResult describing the outcome.
    Never writes ACTIVE records. Never auto-promotes.
    """
    if active_medications is None:
        active_medications = []

    # 1. Safe mode check — queue if enrichment disabled
    if safe_mode:
        qi = queue.enqueue(candidate.safe_candidate_id, "safe_mode_enrichment_disabled")
        return EnrichmentWriteResult(
            action=EnrichmentAction.QUEUE_PENDING_SAFETY,
            status=EnrichmentCandidateStatus.QUEUED_PENDING_SAFETY,
            explanation="Safe mode active — enrichment disabled.",
            ledger_event_ready=False,
            queued_item=qi,
            safe_public_summary={
                "safe_candidate_id": candidate.safe_candidate_id,
                "action": EnrichmentAction.QUEUE_PENDING_SAFETY.value,
                "status": EnrichmentCandidateStatus.QUEUED_PENDING_SAFETY.value,
                "queue_reason": "safe_mode_enrichment_disabled",
                "synthetic": True,
            },
        )

    # 2. Deduplication check against hypothesis + active records
    existing = store.list_hypothesis() + store.list_active()
    if is_duplicate_candidate(candidate, existing):
        return EnrichmentWriteResult(
            action=EnrichmentAction.DISCARD_DUPLICATE,
            status=EnrichmentCandidateStatus.DUPLICATE_DISCARDED,
            explanation="Duplicate candidate discarded — identical record already exists.",
            ledger_event_ready=False,
            safe_public_summary={
                "safe_candidate_id": candidate.safe_candidate_id,
                "action": EnrichmentAction.DISCARD_DUPLICATE.value,
                "status": EnrichmentCandidateStatus.DUPLICATE_DISCARDED.value,
                "synthetic": True,
            },
        )

    # 3. Hypothesis write (includes conflict check + medication safety gate internally)
    result = write_candidate_as_hypothesis(
        candidate,
        store,
        ddi_mode=ddi_mode,
        active_medications=active_medications,
        user_acknowledged_medium=user_acknowledged_medium,
        user_confirmed_high=user_confirmed_high,
    )

    # 4. Queue items for safety-blocked or queued results
    if result.status in (
        EnrichmentCandidateStatus.QUEUED_PENDING_SAFETY,
        EnrichmentCandidateStatus.BLOCKED_SAFETY,
    ):
        med_result = result.medication_gate_result
        if med_result is not None:
            ddi_val = med_result.ddi_status.value if hasattr(med_result.ddi_status, "value") else str(med_result.ddi_status)
            if ddi_val == "high_blocked":
                reason = "blocked_high_ddi"
            elif ddi_val in ("medium", "pending"):
                reason = "pending_medium_ack"
            else:
                reason = "pending_ddi_check"
        else:
            reason = "pending_ddi_check"
        qi = queue.enqueue(candidate.safe_candidate_id, reason)
        result.queued_item = qi

    return result
