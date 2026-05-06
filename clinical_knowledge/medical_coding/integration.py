"""Integration helpers for CKA-B07 Medical Coding.

Provides:
- code_entity()        — query terminology sources in order
- coding_candidate_from_mkb_record()  — build CodingCandidate from MKBRecord
- apply_coding_result_to_record()     — attach coding metadata to MKBRecord

Rules:
- Never promote hypothesis to active.
- Never clear or override DDI status.
- Never bypass Truth Resolution or Medication Safety Gate.
- Never call external APIs.
- Never hallucinate codes.
- Coding is metadata normalisation only.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from clinical_knowledge.medical_coding.models import (
    CodingCandidate,
    CodingResult,
    CodingStatus,
    TerminologySourceStatus,
)
from clinical_knowledge.medical_coding.terminology_source import TerminologySource

if TYPE_CHECKING:
    from clinical_knowledge.models import MKBRecord
    from clinical_knowledge.store import MKBStore

_SALT = "medai_cka_b07_integration_v1"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().strip())


def _safe_candidate_id(record_id: str) -> str:
    digest = hashlib.sha256(f"{_SALT}:cand:{record_id}".encode()).hexdigest()[:16]
    return f"cka_cc_{digest}"


# ---------------------------------------------------------------------------
# Coding service
# ---------------------------------------------------------------------------

def code_entity(
    candidate: CodingCandidate,
    terminology_sources: List[TerminologySource],
) -> CodingResult:
    """Query terminology sources in order; return first coded result.

    Behaviour:
    - Normalise entity text deterministically.
    - Query sources left-to-right; return first non-unmapped result.
    - If all sources unavailable: status=coding_unavailable.
    - If no source has a match: status=unmapped.
    - Never invent codes. No external API calls. No LLM calls.
    """
    if not terminology_sources:
        return CodingResult(
            candidate_safe_id=candidate.safe_candidate_id,
            status=CodingStatus.CODING_UNAVAILABLE,
            codes=[],
            preferred_code=None,
            ambiguity_count=0,
            confidence=0.0,
            terminology_source_status=TerminologySourceStatus.UNAVAILABLE,
            explanation="No terminology sources configured.",
            no_code_hallucinated=True,
        )

    all_unavailable = True
    last_unmapped: Optional[CodingResult] = None

    for source in terminology_sources:
        src_status = source.status()
        if src_status == TerminologySourceStatus.UNAVAILABLE:
            continue
        all_unavailable = False

        result = source.lookup(
            candidate.normalized_text,
            fact_type=candidate.fact_type,
            specialty=candidate.specialty,
        )
        # Stamp with correct candidate safe ID
        result.candidate_safe_id = candidate.safe_candidate_id
        result.safe_public_summary["candidate_safe_id"] = candidate.safe_candidate_id

        if result.status == CodingStatus.CODED:
            return result
        if result.status == CodingStatus.AMBIGUOUS:
            return result
        if result.status in (CodingStatus.SOURCE_UNAVAILABLE,):
            continue
        # UNMAPPED — record and continue to next source
        last_unmapped = result

    if all_unavailable:
        return CodingResult(
            candidate_safe_id=candidate.safe_candidate_id,
            status=CodingStatus.CODING_UNAVAILABLE,
            codes=[],
            preferred_code=None,
            ambiguity_count=0,
            confidence=0.0,
            terminology_source_status=TerminologySourceStatus.UNAVAILABLE,
            explanation="All terminology sources unavailable.",
            no_code_hallucinated=True,
        )

    # At least one source was available but no match found
    return last_unmapped or CodingResult(
        candidate_safe_id=candidate.safe_candidate_id,
        status=CodingStatus.UNMAPPED,
        codes=[],
        preferred_code=None,
        ambiguity_count=0,
        confidence=0.0,
        terminology_source_status=TerminologySourceStatus.STUB_ONLY,
        explanation="Entity not found in any available terminology source.",
        no_code_hallucinated=True,
    )


# ---------------------------------------------------------------------------
# MKB record helpers
# ---------------------------------------------------------------------------

def coding_candidate_from_mkb_record(record: "MKBRecord") -> CodingCandidate:
    """Build a CodingCandidate from an MKBRecord.

    Uses safe IDs only — raw record_id is never exposed in public summary.
    """
    rid = getattr(record, "record_id", "") or ""
    safe_cid = _safe_candidate_id(rid)
    entity = getattr(record, "entity_text", "") or ""
    fact_type = getattr(record, "fact_type", "") or ""
    specialty = getattr(record, "specialty", "general") or "general"
    tier = getattr(record, "tier", None)
    tier_val = tier.value if hasattr(tier, "value") else str(tier)

    structured = getattr(record, "structured", {}) or {}
    if isinstance(structured, str):
        import json as _json
        try:
            structured = _json.loads(structured)
        except Exception:
            structured = {}

    return CodingCandidate(
        candidate_id=rid,           # internal only
        safe_candidate_id=safe_cid,
        fact_type=fact_type,
        entity_text=entity,
        normalized_text=_normalize(entity),
        specialty=specialty,
        structured=structured,
        source_record_id=rid,       # internal only — not in public summary
        source_tier=tier_val,
    )


def apply_coding_result_to_record(
    record: "MKBRecord",
    coding_result: CodingResult,
) -> "MKBRecord":
    """Attach coding metadata to an MKBRecord without changing tier/status.

    Rules enforced:
    - tier is NEVER changed (hypothesis stays hypothesis, active stays active).
    - status is NEVER changed.
    - DDI status is NEVER cleared or overridden.
    - No clinical advice added.

    The coding result is stored in record.structured["coding"].
    Returns the mutated record (same object).
    """
    structured = getattr(record, "structured", {}) or {}
    if isinstance(structured, str):
        import json as _json
        try:
            structured = _json.loads(structured)
        except Exception:
            structured = {}
    if not isinstance(structured, dict):
        structured = {}

    status_val = (
        coding_result.status.value
        if hasattr(coding_result.status, "value")
        else str(coding_result.status)
    )

    coding_meta: Dict[str, Any] = {
        "coding_status": status_val,
        "no_code_hallucinated": coding_result.no_code_hallucinated,
        "confidence": coding_result.confidence,
        "preferred_code": (
            coding_result.preferred_code.safe_public_summary
            if coding_result.preferred_code else None
        ),
        "ambiguity_count": coding_result.ambiguity_count,
    }

    structured["coding"] = coding_meta

    # Mutate record — use object.__setattr__ for frozen dataclasses, fallback otherwise
    try:
        object.__setattr__(record, "structured", structured)
    except (AttributeError, TypeError):
        try:
            record.structured = structured
        except AttributeError:
            pass

    # Explicitly do NOT touch tier, status, ddi_status, ddi_checked, ddi_findings
    return record


def write_coding_ledger_event(
    record: "MKBRecord",
    coding_result: CodingResult,
    store: "MKBStore",
    systems_attempted: Optional[List[str]] = None,
) -> None:
    """Write a MEDICAL_CODING ledger event to the store."""
    from clinical_knowledge.ledger import make_medical_coding_event

    rid = getattr(record, "record_id", "") or ""
    safe_id = getattr(record, "safe_record_id", "") or ""
    status_val = (
        coding_result.status.value
        if hasattr(coding_result.status, "value")
        else str(coding_result.status)
    )
    evt = make_medical_coding_event(
        record_id=rid,
        safe_record_id=safe_id,
        coding_status=status_val,
        systems_attempted=systems_attempted or [],
        preferred_code_summary=(
            coding_result.preferred_code.safe_public_summary
            if coding_result.preferred_code else None
        ),
    )
    store.append_ledger_event(evt)
