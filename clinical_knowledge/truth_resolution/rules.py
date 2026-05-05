"""CKA-B04 Truth Resolution ordered priority rules.

7 rules executed in order. First matching rule wins.
No rule skipping.
"""
from __future__ import annotations

import copy
import json as _json
from datetime import datetime, timezone
from typing import Optional

from clinical_knowledge.models import KnowledgeTier, RecordStatus, TrustLevel
from clinical_knowledge.truth_resolution.models import (
    ConflictPair,
    ConflictType,
    ResolutionAction,
    ResolutionRule,
    TruthResolutionResult,
)

_RECENCY_DAYS_THRESHOLD = 90


def _parse_dt(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _days_apart(a_ts: Optional[str], b_ts: Optional[str]) -> Optional[float]:
    da = _parse_dt(a_ts)
    db = _parse_dt(b_ts)
    if da is None or db is None:
        return None
    return abs((da - db).total_seconds()) / 86400.0


def _trust(record) -> int:
    tl = getattr(record, "trust_level", None)
    if tl is None:
        return 5
    if isinstance(tl, TrustLevel):
        return tl.value
    return int(tl)


def _get_structured(record) -> dict:
    s = getattr(record, "structured", None) or {}
    if isinstance(s, str):
        try:
            s = _json.loads(s)
        except (ValueError, TypeError):
            s = {}
    return s if isinstance(s, dict) else {}


def _safe_id(record) -> str:
    return getattr(record, "safe_record_id", "") or ""


def _record_id(record) -> str:
    return getattr(record, "record_id", "") or ""


def _public_summary(rule: ResolutionRule, action: ResolutionAction, pair: ConflictPair) -> dict:
    return {
        "rule_applied": rule.value,
        "resolution": action.value,
        "winner_safe_id": _safe_id(pair.candidate_fact)
        if action == ResolutionAction.REPLACE_WITH_NEW
        else _safe_id(pair.existing_fact),
        "loser_safe_id": _safe_id(pair.existing_fact)
        if action != ResolutionAction.REPLACE_WITH_NEW
        else _safe_id(pair.candidate_fact),
        "conflict_type": pair.conflict_type.value,
    }


# ---------------------------------------------------------------------------
# Rule 1 — Clinical supremacy (trust_level=1 beats all)
# ---------------------------------------------------------------------------

def rule_clinical_supremacy(pair: ConflictPair) -> Optional[TruthResolutionResult]:
    cand = pair.candidate_fact
    exist = pair.existing_fact
    tc = _trust(cand)
    te = _trust(exist)

    if tc == 1 and te > 1:
        winner, loser = cand, exist
        action = ResolutionAction.REPLACE_WITH_NEW
    elif te == 1 and tc > 1:
        winner, loser = exist, cand
        action = ResolutionAction.KEEP_EXISTING
    else:
        return None

    return TruthResolutionResult(
        resolution=action,
        rule_applied=ResolutionRule.CLINICAL_SUPREMACY,
        winner=winner,
        loser_id=_record_id(loser),
        merged_record=None,
        quarantined_record_ids=[],
        superseded_record_ids=[_record_id(loser)],
        confidence=0.95,
        explanation=(
            f"Clinical supremacy: trust_level=1 record takes precedence. "
            f"Superseded record {_safe_id(loser)}."
        ),
        requires_review=False,
        safe_public_summary=_public_summary(
            ResolutionRule.CLINICAL_SUPREMACY, action, pair
        ),
    )


# ---------------------------------------------------------------------------
# Rule 2 — Peer review beats AI
# ---------------------------------------------------------------------------

def rule_peer_review_beats_ai(pair: ConflictPair) -> Optional[TruthResolutionResult]:
    cand = pair.candidate_fact
    exist = pair.existing_fact
    tc = _trust(cand)
    te = _trust(exist)

    if tc == 2 and te >= 3:
        winner, loser = cand, exist
        action = ResolutionAction.REPLACE_WITH_NEW
    elif te == 2 and tc >= 3:
        winner, loser = exist, cand
        action = ResolutionAction.KEEP_EXISTING
    else:
        return None

    return TruthResolutionResult(
        resolution=action,
        rule_applied=ResolutionRule.PEER_REVIEW_BEATS_AI,
        winner=winner,
        loser_id=_record_id(loser),
        merged_record=None,
        quarantined_record_ids=[],
        superseded_record_ids=[_record_id(loser)],
        confidence=0.90,
        explanation=(
            f"Peer review (trust_level=2) supersedes AI-derived record. "
            f"Superseded: {_safe_id(loser)}."
        ),
        requires_review=False,
        safe_public_summary=_public_summary(
            ResolutionRule.PEER_REVIEW_BEATS_AI, action, pair
        ),
    )


# ---------------------------------------------------------------------------
# Rule 3 — Recency same trust (> 90 days apart)
# ---------------------------------------------------------------------------

def rule_recency_same_trust(pair: ConflictPair) -> Optional[TruthResolutionResult]:
    cand = pair.candidate_fact
    exist = pair.existing_fact

    if _trust(cand) != _trust(exist):
        return None

    cand_ts = getattr(cand, "created_at", None)
    exist_ts = getattr(exist, "created_at", None)
    days = _days_apart(cand_ts, exist_ts)

    if days is None or days <= _RECENCY_DAYS_THRESHOLD:
        return None

    cand_dt = _parse_dt(cand_ts)
    exist_dt = _parse_dt(exist_ts)

    if cand_dt >= exist_dt:
        winner, loser = cand, exist
        action = ResolutionAction.REPLACE_WITH_NEW
    else:
        winner, loser = exist, cand
        action = ResolutionAction.KEEP_EXISTING

    return TruthResolutionResult(
        resolution=action,
        rule_applied=ResolutionRule.RECENCY_SAME_TRUST,
        winner=winner,
        loser_id=_record_id(loser),
        merged_record=None,
        quarantined_record_ids=[],
        superseded_record_ids=[_record_id(loser)],
        confidence=0.80,
        explanation=(
            f"Recency rule: newer record ({_safe_id(winner)}) supersedes older "
            f"({_safe_id(loser)}) by {days:.0f} days at same trust level."
        ),
        requires_review=False,
        safe_public_summary=_public_summary(
            ResolutionRule.RECENCY_SAME_TRUST, action, pair
        ),
    )


# ---------------------------------------------------------------------------
# Rule 4 — Source agreement (candidate has more sources)
# ---------------------------------------------------------------------------

def rule_source_agreement(pair: ConflictPair) -> Optional[TruthResolutionResult]:
    cand = pair.candidate_fact
    exist = pair.existing_fact

    cand_count = _get_structured(cand).get("source_count", 0)
    exist_count = _get_structured(exist).get("source_count", 0)

    if cand_count < 2:
        return None
    if cand_count <= exist_count:
        return None

    return TruthResolutionResult(
        resolution=ResolutionAction.REPLACE_WITH_NEW,
        rule_applied=ResolutionRule.SOURCE_AGREEMENT,
        winner=cand,
        loser_id=_record_id(exist),
        merged_record=None,
        quarantined_record_ids=[],
        superseded_record_ids=[_record_id(exist)],
        confidence=0.75,
        explanation=(
            f"Source agreement: candidate has {cand_count} sources vs "
            f"{exist_count}. Superseded: {_safe_id(exist)}."
        ),
        requires_review=False,
        safe_public_summary=_public_summary(
            ResolutionRule.SOURCE_AGREEMENT, ResolutionAction.REPLACE_WITH_NEW, pair
        ),
    )


# ---------------------------------------------------------------------------
# Rule 5 — Value range merge (numeric values, same week)
# ---------------------------------------------------------------------------

def _numeric_value(record) -> Optional[float]:
    struct = _get_structured(record)
    val = struct.get("value")
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _same_week(ts_a: Optional[str], ts_b: Optional[str]) -> bool:
    da = _parse_dt(ts_a)
    db = _parse_dt(ts_b)
    if da is None or db is None:
        return True  # no dates → assume same period
    return abs((da - db).total_seconds()) <= 7 * 86400


def rule_value_range_merge(pair: ConflictPair) -> Optional[TruthResolutionResult]:
    if pair.conflict_type not in (
        ConflictType.VALUE_CONFLICT, ConflictType.UNKNOWN_CONFLICT
    ):
        return None

    cand = pair.candidate_fact
    exist = pair.existing_fact

    cv = _numeric_value(cand)
    ev = _numeric_value(exist)

    if cv is None or ev is None:
        return None

    cand_ts = getattr(cand, "created_at", None)
    exist_ts = getattr(exist, "created_at", None)
    if not _same_week(cand_ts, exist_ts):
        return None

    merged = copy.deepcopy(cand)
    lo, hi = min(cv, ev), max(cv, ev)
    struct = dict(_get_structured(merged))
    struct["value"] = f"{lo}-{hi}"
    struct["value_range"] = {"low": lo, "high": hi}
    struct["merged_from_safe_ids"] = [_safe_id(cand), _safe_id(exist)]
    struct.pop("source_ref", None)  # never carry raw source refs into merged record
    object.__setattr__(merged, "structured", struct) if hasattr(merged, "__dataclass_fields__") else setattr(merged, "structured", struct)

    return TruthResolutionResult(
        resolution=ResolutionAction.MERGE,
        rule_applied=ResolutionRule.VALUE_RANGE_MERGE,
        winner=merged,
        loser_id=_record_id(exist),
        merged_record=merged,
        quarantined_record_ids=[],
        superseded_record_ids=[_record_id(cand), _record_id(exist)],
        confidence=0.70,
        explanation=(
            f"Value range merge: numeric values {cv} and {ev} merged into "
            f"range {lo}-{hi} for same measurement period."
        ),
        requires_review=False,
        safe_public_summary={
            "rule_applied": ResolutionRule.VALUE_RANGE_MERGE.value,
            "resolution": ResolutionAction.MERGE.value,
            "value_range": {"low": lo, "high": hi},
            "merged_safe_ids": [_safe_id(cand), _safe_id(exist)],
        },
    )


# ---------------------------------------------------------------------------
# Rule 6 — Medication dose conflict (quarantine both)
# ---------------------------------------------------------------------------

def rule_medication_dose_conflict(pair: ConflictPair) -> Optional[TruthResolutionResult]:
    if pair.conflict_type != ConflictType.MEDICATION_DOSE_CONFLICT:
        return None

    cand = pair.candidate_fact
    exist = pair.existing_fact

    return TruthResolutionResult(
        resolution=ResolutionAction.QUARANTINE,
        rule_applied=ResolutionRule.MEDICATION_DOSE_CONFLICT,
        winner=None,
        loser_id=None,
        merged_record=None,
        quarantined_record_ids=[_record_id(cand), _record_id(exist)],
        superseded_record_ids=[],
        confidence=0.0,
        explanation=(
            "Conflicting medication doses require qualified clinician verification."
        ),
        requires_review=True,
        safe_public_summary={
            "rule_applied": ResolutionRule.MEDICATION_DOSE_CONFLICT.value,
            "resolution": ResolutionAction.QUARANTINE.value,
            "quarantined_safe_ids": [_safe_id(cand), _safe_id(exist)],
            "requires_review": True,
        },
    )


# ---------------------------------------------------------------------------
# Rule 7 — Unresolvable (quarantine candidate)
# ---------------------------------------------------------------------------

def rule_unresolvable(pair: ConflictPair) -> TruthResolutionResult:
    cand = pair.candidate_fact

    return TruthResolutionResult(
        resolution=ResolutionAction.QUARANTINE,
        rule_applied=ResolutionRule.UNRESOLVABLE,
        winner=None,
        loser_id=None,
        merged_record=None,
        quarantined_record_ids=[_record_id(cand)],
        superseded_record_ids=[],
        confidence=0.0,
        explanation=(
            f"Conflict unresolvable by ordered rules. "
            f"Candidate {_safe_id(cand)} quarantined pending review."
        ),
        requires_review=True,
        safe_public_summary={
            "rule_applied": ResolutionRule.UNRESOLVABLE.value,
            "resolution": ResolutionAction.QUARANTINE.value,
            "quarantined_safe_ids": [_safe_id(cand)],
            "requires_review": True,
        },
    )


# Ordered rule chain
ORDERED_RULES = [
    rule_clinical_supremacy,
    rule_peer_review_beats_ai,
    rule_recency_same_trust,
    rule_source_agreement,
    rule_value_range_merge,
    rule_medication_dose_conflict,
]
