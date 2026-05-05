"""Data models for CKA-B04 Truth Resolution Engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ConflictType(str, Enum):
    VALUE_CONFLICT = "value_conflict"
    STATUS_CONFLICT = "status_conflict"
    DATE_CONFLICT = "date_conflict"
    SOURCE_CONFLICT = "source_conflict"
    MEDICATION_DOSE_CONFLICT = "medication_dose_conflict"
    UNKNOWN_CONFLICT = "unknown_conflict"


class ResolutionAction(str, Enum):
    KEEP_EXISTING = "keep_existing"
    REPLACE_WITH_NEW = "replace_with_new"
    MERGE = "merge"
    QUARANTINE = "quarantine"


class ResolutionRule(str, Enum):
    CLINICAL_SUPREMACY = "clinical_supremacy"
    PEER_REVIEW_BEATS_AI = "peer_review_beats_ai"
    RECENCY_SAME_TRUST = "recency_same_trust"
    SOURCE_AGREEMENT = "source_agreement"
    VALUE_RANGE_MERGE = "value_range_merge"
    MEDICATION_DOSE_CONFLICT = "medication_dose_conflict"
    UNRESOLVABLE = "unresolvable"


@dataclass
class ConflictPair:
    candidate_fact: Any        # MKBRecord (typed as Any to avoid circular import)
    existing_fact: Any
    conflict_type: ConflictType
    detected_reasons: List[str]
    safe_public_summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TruthResolutionResult:
    resolution: ResolutionAction
    rule_applied: ResolutionRule
    winner: Optional[Any]              # MKBRecord or None
    loser_id: Optional[str]            # record_id of loser
    merged_record: Optional[Any]       # MKBRecord if merge; None otherwise
    quarantined_record_ids: List[str]
    superseded_record_ids: List[str]
    confidence: float
    explanation: str
    requires_review: bool
    ledger_event_ready: bool = True
    safe_public_summary: Dict[str, Any] = field(default_factory=dict)
