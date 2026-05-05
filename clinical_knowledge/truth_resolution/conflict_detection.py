"""Deterministic conflict detection for CKA-B04.

No external NLP. No clinical inference. Lowercase text normalization only.
"""
from __future__ import annotations

import re
from typing import Optional

from clinical_knowledge.truth_resolution.models import ConflictPair, ConflictType

# Medication keywords used to identify medication-type facts
_MED_FACT_TYPES = {
    "medication", "drug", "prescription", "antiepileptic", "anticonvulsant",
    "pharmaceutical", "dose", "dosage",
}

_DOSE_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|ml|g|units?|iu)\b", re.I)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().strip())


def _token_overlap(a: str, b: str) -> float:
    ta = set(_normalize(a).split())
    tb = set(_normalize(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def _is_medication_fact(record) -> bool:
    ft = _normalize(getattr(record, "fact_type", "") or "")
    return any(kw in ft for kw in _MED_FACT_TYPES)


def _med_name(text: str) -> str:
    """Extract medication name by stripping dose and frequency words."""
    cleaned = _DOSE_PATTERN.sub("", text)
    freq_words = re.compile(
        r"\b(?:once|twice|three\s+times|daily|weekly|hourly|per\s+day|bd|tds|qds|od|tid|bid|prn)\b",
        re.I,
    )
    cleaned = freq_words.sub("", cleaned)
    return _normalize(cleaned)


def _same_entity(a, b) -> bool:
    na = _normalize(getattr(a, "entity_text", "") or "")
    nb = _normalize(getattr(b, "entity_text", "") or "")
    if na == nb:
        return True
    # For medication facts, compare name without dose/frequency
    if _is_medication_fact(a) and _is_medication_fact(b):
        ma = _med_name(na)
        mb = _med_name(nb)
        if ma and mb and _token_overlap(ma, mb) >= 0.5:
            return True
    overlap = _token_overlap(na, nb)
    return overlap >= 0.6


def _same_fact_type(a, b) -> bool:
    return _normalize(getattr(a, "fact_type", "") or "") == _normalize(
        getattr(b, "fact_type", "") or ""
    )


def _extract_dose(text: str) -> Optional[str]:
    m = _DOSE_PATTERN.search(text or "")
    return m.group(0).lower() if m else None


def _safe_public_summary(candidate, existing, conflict_type: ConflictType) -> dict:
    return {
        "candidate_safe_id": getattr(candidate, "safe_record_id", ""),
        "existing_safe_id": getattr(existing, "safe_record_id", ""),
        "conflict_type": conflict_type.value,
        "fact_type": getattr(candidate, "fact_type", ""),
        "specialty": getattr(candidate, "specialty", "general"),
    }


def _get_structured(record) -> dict:
    """Get structured dict, handling both dict and JSON string forms."""
    import json as _json
    s = getattr(record, "structured", None) or {}
    if isinstance(s, str):
        try:
            s = _json.loads(s)
        except (ValueError, TypeError):
            s = {}
    return s if isinstance(s, dict) else {}


def detect_conflict(candidate, existing) -> Optional[ConflictPair]:
    """Detect conflict between candidate and existing MKBRecord.

    Returns ConflictPair if a conflict is found, None if no conflict.
    """
    if not _same_fact_type(candidate, existing):
        return None
    if not _same_entity(candidate, existing):
        return None

    reasons = []

    # Medication dose conflict — check first (most specific)
    if _is_medication_fact(candidate) and _is_medication_fact(existing):
        cand_dose = _extract_dose(getattr(candidate, "entity_text", "") or "")
        exist_dose = _extract_dose(getattr(existing, "entity_text", "") or "")
        if cand_dose and exist_dose and cand_dose != exist_dose:
            reasons.append(
                f"Medication dose mismatch: candidate={cand_dose}, existing={exist_dose}"
            )
            return ConflictPair(
                candidate_fact=candidate,
                existing_fact=existing,
                conflict_type=ConflictType.MEDICATION_DOSE_CONFLICT,
                detected_reasons=reasons,
                safe_public_summary=_safe_public_summary(
                    candidate, existing, ConflictType.MEDICATION_DOSE_CONFLICT
                ),
            )

    # Value conflict — structured values differ
    cand_struct = _get_structured(candidate)
    exist_struct = _get_structured(existing)
    if cand_struct and exist_struct:
        cand_val = cand_struct.get("value")
        exist_val = exist_struct.get("value")
        if cand_val is not None and exist_val is not None and cand_val != exist_val:
            reasons.append(
                f"Structured value mismatch: candidate={cand_val}, existing={exist_val}"
            )
            return ConflictPair(
                candidate_fact=candidate,
                existing_fact=existing,
                conflict_type=ConflictType.VALUE_CONFLICT,
                detected_reasons=reasons,
                safe_public_summary=_safe_public_summary(
                    candidate, existing, ConflictType.VALUE_CONFLICT
                ),
            )

    # Status conflict — both claim active/confirmed
    from clinical_knowledge.models import RecordStatus
    cand_status = getattr(candidate, "status", None)
    exist_status = getattr(existing, "status", None)
    if (
        cand_status == RecordStatus.CONFIRMED
        and exist_status == RecordStatus.CONFIRMED
        and cand_struct.get("status_value") != exist_struct.get("status_value")
        and cand_struct.get("status_value") is not None
    ):
        reasons.append("Both records have conflicting confirmed status values")
        return ConflictPair(
            candidate_fact=candidate,
            existing_fact=existing,
            conflict_type=ConflictType.STATUS_CONFLICT,
            detected_reasons=reasons,
            safe_public_summary=_safe_public_summary(
                candidate, existing, ConflictType.STATUS_CONFLICT
            ),
        )

    # Date conflict — created_at or structured date fields conflict
    cand_date = cand_struct.get("date_value")
    exist_date = exist_struct.get("date_value")
    if cand_date and exist_date and cand_date != exist_date:
        reasons.append(f"Date field mismatch: candidate={cand_date}, existing={exist_date}")
        return ConflictPair(
            candidate_fact=candidate,
            existing_fact=existing,
            conflict_type=ConflictType.DATE_CONFLICT,
            detected_reasons=reasons,
            safe_public_summary=_safe_public_summary(
                candidate, existing, ConflictType.DATE_CONFLICT
            ),
        )

    # Source conflict — incompatible source_type/trust
    from clinical_knowledge.models import SourceType
    cand_src = getattr(candidate, "source_type", None)
    exist_src = getattr(existing, "source_type", None)
    if (
        cand_src == SourceType.STUB_CONNECTOR
        and exist_src == SourceType.OPERATOR_MANUAL
    ):
        reasons.append(
            "Source type conflict: stub connector vs operator manual"
        )
        return ConflictPair(
            candidate_fact=candidate,
            existing_fact=existing,
            conflict_type=ConflictType.SOURCE_CONFLICT,
            detected_reasons=reasons,
            safe_public_summary=_safe_public_summary(
                candidate, existing, ConflictType.SOURCE_CONFLICT
            ),
        )

    # Fallback: same fact_type/entity but no specific conflict detected
    # — consider unknown conflict if structured values differ at all
    if cand_struct != exist_struct and (cand_struct or exist_struct):
        reasons.append("Structured data differs but conflict type is ambiguous")
        return ConflictPair(
            candidate_fact=candidate,
            existing_fact=existing,
            conflict_type=ConflictType.UNKNOWN_CONFLICT,
            detected_reasons=reasons,
            safe_public_summary=_safe_public_summary(
                candidate, existing, ConflictType.UNKNOWN_CONFLICT
            ),
        )

    return None
