"""Hypothesis writer for CKA-B06 Controlled Enrichment.

Converts an EnrichmentCandidate to an MKBRecord with tier=HYPOTHESIS.

Rules:
- AI-derived facts must never be written as ACTIVE.
- Medication candidates must pass through CKA-B05 Medication Safety Gate.
- Conflicts with active records route to CKA-B04 Truth Resolution.
- ENRICH_PROMOTE=False blocks auto-promotion.
- No clinical recommendations generated.
- No prescription dosing advice generated.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import TYPE_CHECKING, List, Optional

from clinical_knowledge.enrichment.models import (
    EnrichmentAction,
    EnrichmentCandidate,
    EnrichmentCandidateStatus,
    EnrichmentWriteResult,
)
from clinical_knowledge.ledger import make_enrichment_write_event
from clinical_knowledge.medication_safety.integration import attempt_medication_record_write
from clinical_knowledge.medication_safety.models import (
    DDICheckStatus,
    MedicationSafetyAction,
)
from clinical_knowledge.models import (
    KnowledgeTier,
    MKBRecord,
    RecordStatus,
    SourceType,
    TrustLevel,
)
from clinical_knowledge.safe_ids import make_safe_record_id, new_record_id
from clinical_knowledge.truth_resolution.conflict_detection import detect_conflict
from clinical_knowledge.truth_resolution.engine import resolve_conflict
from clinical_knowledge.truth_resolution.models import ConflictPair, ResolutionAction

if TYPE_CHECKING:
    from clinical_knowledge.store import MKBStore

_SALT = "medai_cka_b06_writer_v1"

_MEDICATION_FACT_TYPES = frozenset({
    "medication", "drug", "prescription", "antiepileptic", "anticonvulsant",
    "pharmaceutical", "medication_antiepileptic", "medication_reference",
})


def _is_medication_candidate(candidate: EnrichmentCandidate) -> bool:
    ft = (candidate.fact_type or "").lower().strip()
    return ft in _MEDICATION_FACT_TYPES or ft.startswith("medication")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _candidate_to_mkb_record(candidate: EnrichmentCandidate) -> MKBRecord:
    """Convert EnrichmentCandidate to MKBRecord. Always tier=HYPOTHESIS."""
    rid = new_record_id()
    safe_id = make_safe_record_id(rid)

    try:
        tl = TrustLevel(candidate.proposed_trust_level)
    except ValueError:
        tl = TrustLevel.OPERATOR_REVIEWED

    return MKBRecord(
        record_id=rid,
        safe_record_id=safe_id,
        session_id="enrichment_b06",
        fact_type=candidate.fact_type,
        entity_text=candidate.entity_text,
        structured=candidate.structured,
        specialty=candidate.specialty,
        source_type=SourceType.STUB_CONNECTOR,
        source_ref=candidate.source_response_hash,  # always a hash, never raw
        trust_level=tl,
        tier=KnowledgeTier.HYPOTHESIS,  # forced — never ACTIVE for enrichment
        status=RecordStatus.PENDING,
        confidence=candidate.confidence,
        extraction_method=candidate.extraction_method,
        requires_review=True,
    )


def _check_for_active_conflict(
    candidate_record: MKBRecord,
    store: "MKBStore",
) -> Optional[ConflictPair]:
    """Return the first ConflictPair if the candidate conflicts with any active record."""
    active = store.list_active()
    for row in active:
        rid = row.get("record_id", "")
        existing = store.fetch_by_record_id(rid)
        if existing is None:
            continue
        existing_rec = _row_to_mkb(existing)
        pair = detect_conflict(candidate_record, existing_rec)
        if pair is not None:
            return pair
    return None


def _row_to_mkb(row: dict) -> MKBRecord:
    """Reconstruct a minimal MKBRecord from a store dict for conflict detection."""
    import json as _json
    structured = row.get("structured", {})
    if isinstance(structured, str):
        try:
            structured = _json.loads(structured)
        except Exception:
            structured = {}

    try:
        tl = TrustLevel(int(row.get("trust_level", 5)))
    except (ValueError, TypeError):
        tl = TrustLevel.UNVERIFIED

    try:
        tier = KnowledgeTier(row.get("tier", "hypothesis"))
    except ValueError:
        tier = KnowledgeTier.HYPOTHESIS

    try:
        status = RecordStatus(row.get("status", "pending"))
    except ValueError:
        status = RecordStatus.PENDING

    try:
        src = SourceType(row.get("source_type", "synthetic"))
    except ValueError:
        src = SourceType.SYNTHETIC

    return MKBRecord(
        record_id=row.get("record_id", ""),
        safe_record_id=row.get("safe_record_id", ""),
        session_id=row.get("session_id", ""),
        fact_type=row.get("fact_type", ""),
        entity_text=row.get("entity_text", ""),
        structured=structured,
        specialty=row.get("specialty", "general"),
        source_type=src,
        trust_level=tl,
        tier=tier,
        status=status,
        confidence=float(row.get("confidence", 0.0)),
        extraction_method=row.get("extraction_method", "manual"),
        requires_review=bool(row.get("requires_review", 0)),
    )


def write_candidate_as_hypothesis(
    candidate: EnrichmentCandidate,
    store: "MKBStore",
    *,
    ddi_mode: str = "normal",
    active_medications: Optional[List[str]] = None,
    user_acknowledged_medium: bool = False,
    user_confirmed_high: bool = False,
) -> EnrichmentWriteResult:
    """Write candidate as HYPOTHESIS record via full safety pipeline.

    Does NOT perform auto-promotion. Does NOT write ACTIVE records.
    """
    if active_medications is None:
        active_medications = []

    candidate_record = _candidate_to_mkb_record(candidate)

    # --- Conflict check against existing active records ---
    conflict_pair = _check_for_active_conflict(candidate_record, store)
    if conflict_pair is not None:
        tr_result = resolve_conflict(conflict_pair)
        if tr_result.resolution == ResolutionAction.QUARANTINE:
            return EnrichmentWriteResult(
                action=EnrichmentAction.ROUTE_TRUTH_RESOLUTION,
                status=EnrichmentCandidateStatus.CONFLICT_QUARANTINED,
                explanation=(
                    "Candidate conflicts with existing active record. "
                    "Truth Resolution quarantine applied."
                ),
                ledger_event_ready=True,
                truth_resolution_result=tr_result,
                safe_public_summary={
                    "safe_candidate_id": candidate.safe_candidate_id,
                    "action": EnrichmentAction.ROUTE_TRUTH_RESOLUTION.value,
                    "status": EnrichmentCandidateStatus.CONFLICT_QUARANTINED.value,
                    "tr_resolution": tr_result.resolution.value,
                    "synthetic": True,
                },
            )
        # Non-quarantine resolution: still write only as hypothesis — no active promotion
        # Fall through to write the candidate as hypothesis

    # --- Medication Safety Gate ---
    if _is_medication_candidate(candidate):
        med_result = attempt_medication_record_write(
            candidate_record,
            store,
            ddi_mode=ddi_mode,
            active_medications=active_medications,
            user_acknowledged=user_acknowledged_medium,
            user_confirmed_high=user_confirmed_high,
        )
        if not med_result.allowed_to_write:
            ddi_st = med_result.ddi_status
            if ddi_st == DDICheckStatus.HIGH_BLOCKED or \
               med_result.action == MedicationSafetyAction.BLOCK_REQUIRES_CONFIRMATION:
                return EnrichmentWriteResult(
                    action=EnrichmentAction.BLOCK_SAFETY,
                    status=EnrichmentCandidateStatus.BLOCKED_SAFETY,
                    explanation=(
                        "Medication candidate blocked: HIGH DDI severity. "
                        "Explicit user confirmation required before any write."
                    ),
                    ledger_event_ready=True,
                    medication_gate_result=med_result,
                    safe_public_summary={
                        "safe_candidate_id": candidate.safe_candidate_id,
                        "action": EnrichmentAction.BLOCK_SAFETY.value,
                        "status": EnrichmentCandidateStatus.BLOCKED_SAFETY.value,
                        "ddi_status": ddi_st.value,
                        "synthetic": True,
                    },
                )
            if ddi_st in (DDICheckStatus.PENDING, DDICheckStatus.MEDIUM):
                return EnrichmentWriteResult(
                    action=EnrichmentAction.QUEUE_PENDING_SAFETY,
                    status=EnrichmentCandidateStatus.QUEUED_PENDING_SAFETY,
                    explanation=(
                        "Medication candidate queued: DDI check pending or MEDIUM "
                        "severity requires acknowledgment."
                    ),
                    ledger_event_ready=True,
                    medication_gate_result=med_result,
                    safe_public_summary={
                        "safe_candidate_id": candidate.safe_candidate_id,
                        "action": EnrichmentAction.QUEUE_PENDING_SAFETY.value,
                        "status": EnrichmentCandidateStatus.QUEUED_PENDING_SAFETY.value,
                        "ddi_status": ddi_st.value,
                        "synthetic": True,
                    },
                )
            # UNAVAILABLE
            return EnrichmentWriteResult(
                action=EnrichmentAction.QUEUE_PENDING_SAFETY,
                status=EnrichmentCandidateStatus.QUEUED_PENDING_SAFETY,
                explanation="DDI unavailable — medication hypothesis write queued pending safety check.",
                ledger_event_ready=True,
                medication_gate_result=med_result,
                safe_public_summary={
                    "safe_candidate_id": candidate.safe_candidate_id,
                    "action": EnrichmentAction.QUEUE_PENDING_SAFETY.value,
                    "status": EnrichmentCandidateStatus.QUEUED_PENDING_SAFETY.value,
                    "ddi_status": ddi_st.value,
                    "synthetic": True,
                },
            )
        # DDI allowed — retain gate result for metadata
        ddi_meta = med_result
        # Record was already inserted by attempt_medication_record_write
        _write_enrichment_ledger(candidate_record, store, candidate.safe_candidate_id)
        return EnrichmentWriteResult(
            action=EnrichmentAction.WRITE_HYPOTHESIS,
            status=EnrichmentCandidateStatus.WRITTEN_HYPOTHESIS,
            explanation="Medication hypothesis written; DDI cleared/noted.",
            ledger_event_ready=True,
            written_record=candidate_record,
            medication_gate_result=ddi_meta,
            safe_public_summary={
                "safe_candidate_id": candidate.safe_candidate_id,
                "safe_record_id": candidate_record.safe_record_id,
                "action": EnrichmentAction.WRITE_HYPOTHESIS.value,
                "status": EnrichmentCandidateStatus.WRITTEN_HYPOTHESIS.value,
                "tier": "hypothesis",
                "ddi_status": ddi_meta.ddi_status.value,
                "synthetic": True,
            },
        )

    # --- Non-medication: write hypothesis directly ---
    store.insert_record(candidate_record)
    _write_enrichment_ledger(candidate_record, store, candidate.safe_candidate_id)

    return EnrichmentWriteResult(
        action=EnrichmentAction.WRITE_HYPOTHESIS,
        status=EnrichmentCandidateStatus.WRITTEN_HYPOTHESIS,
        explanation="Candidate written as HYPOTHESIS.",
        ledger_event_ready=True,
        written_record=candidate_record,
        safe_public_summary={
            "safe_candidate_id": candidate.safe_candidate_id,
            "safe_record_id": candidate_record.safe_record_id,
            "action": EnrichmentAction.WRITE_HYPOTHESIS.value,
            "status": EnrichmentCandidateStatus.WRITTEN_HYPOTHESIS.value,
            "tier": "hypothesis",
            "synthetic": True,
        },
    )


def _write_enrichment_ledger(record: MKBRecord, store: "MKBStore", safe_candidate_id: str) -> None:
    evt = make_enrichment_write_event(
        record_id=record.record_id,
        safe_record_id=record.safe_record_id,
        tier=record.tier.value if hasattr(record.tier, "value") else str(record.tier),
        trust_level=record.trust_level.value if hasattr(record.trust_level, "value") else int(record.trust_level),
        safe_candidate_id=safe_candidate_id,
    )
    store.append_ledger_event(evt)
