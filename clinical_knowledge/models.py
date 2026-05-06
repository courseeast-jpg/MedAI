"""Core data models for CKA-B01 — MKBRecord, LedgerEvent, enums.

Uses dataclasses (no external deps beyond stdlib + optional pydantic).
Pydantic not required here; dataclasses + __post_init__ validation suffices.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class KnowledgeTier(str, Enum):
    ACTIVE = "active"
    HYPOTHESIS = "hypothesis"
    QUARANTINED = "quarantined"
    SUPERSEDED = "superseded"


class RecordStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    ARCHIVED = "archived"


class TrustLevel(int, Enum):
    EXPERT_VALIDATED = 1       # highest trust
    PEER_REVIEWED = 2
    OPERATOR_REVIEWED = 3
    MODEL_SUGGESTED = 4
    UNVERIFIED = 5             # lowest trust


class SourceType(str, Enum):
    OPERATOR_MANUAL = "operator_manual"
    EXTRACTION_PIPELINE = "extraction_pipeline"
    STUB_CONNECTOR = "stub_connector"
    SYNTHETIC = "synthetic"
    UNKNOWN = "unknown"


class DDIStatus(str, Enum):
    NOT_CHECKED = "not_checked"
    CLEAR = "clear"
    WARNING = "warning"
    BLOCKED = "blocked"


# Ledger event types active in CKA-B01
class LedgerEventType(str, Enum):
    MKB_RECORD_CREATED = "mkb_record_created"
    MKB_RECORD_UPDATED = "mkb_record_updated"
    TIER_CHANGED = "tier_changed"
    STATUS_CHANGED = "status_changed"
    VALIDATION_RUN = "validation_run"
    PRIVACY_AUDIT = "privacy_audit"     # CKA-B02: privacy boundary audit event
    # Reserved — not active behavior in B01:
    TRUTH_RESOLUTION = "truth_resolution"
    QUARANTINE = "quarantine"
    DDI_BLOCK = "ddi_block"
    DDI_WARNING = "ddi_warning"
    ENRICHMENT_WRITE = "enrichment_write"
    HYPOTHESIS_PROMOTED = "hypothesis_promoted"
    MEDICAL_CODING = "medical_coding"
    SAFE_MODE_ENTRY = "safe_mode_entry"
    RESPONSE_DISCARDED = "response_discarded"
    CONNECTOR_EXECUTION = "connector_execution"   # CKA-B08
    CONSENSUS_RESULT = "consensus_result"         # CKA-B08


_RESERVED_EVENT_TYPES: set = set()  # All event types activated through CKA-B08


def _default_tier_for_trust(trust_level: TrustLevel) -> KnowledgeTier:
    """trust_level 1-2 → active; 3-5 → hypothesis."""
    if trust_level in (TrustLevel.EXPERT_VALIDATED, TrustLevel.PEER_REVIEWED):
        return KnowledgeTier.ACTIVE
    return KnowledgeTier.HYPOTHESIS


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# MKBRecord
# ---------------------------------------------------------------------------


@dataclass
class MKBRecord:
    # Identity
    record_id: str
    safe_record_id: str
    session_id: str

    # Clinical content
    fact_type: str
    entity_text: str
    structured: Dict[str, Any] = field(default_factory=dict)
    specialty: str = "general"

    # Provenance
    source_type: SourceType = SourceType.SYNTHETIC
    source_ref: str = ""          # always hashed/safe — never raw path/filename

    # Trust & tier
    trust_level: TrustLevel = TrustLevel.UNVERIFIED
    tier: Optional[KnowledgeTier] = None    # None → auto-assigned from trust_level
    status: RecordStatus = RecordStatus.PENDING
    confidence: float = 0.0

    # Timestamps
    created_at: str = field(default_factory=_now_utc)
    updated_at: str = field(default_factory=_now_utc)

    # DDI
    ddi_checked: bool = False
    ddi_status: DDIStatus = DDIStatus.NOT_CHECKED
    ddi_findings: List[str] = field(default_factory=list)

    # Processing
    extraction_method: str = "manual"
    resolution_id: Optional[str] = None
    promotion_history: List[str] = field(default_factory=list)

    # Review flag — forced True for quarantined/superseded
    requires_review: bool = False

    def __post_init__(self) -> None:
        # Auto-assign tier from trust_level if not explicitly set
        if self.tier is None:
            self.tier = _default_tier_for_trust(self.trust_level)

        # Quarantined/superseded always require review
        if self.tier in (KnowledgeTier.QUARANTINED, KnowledgeTier.SUPERSEDED):
            self.requires_review = True

        # Validate confidence range
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    def is_retrievable_active(self) -> bool:
        """True only if tier is ACTIVE (quarantined/superseded excluded)."""
        return self.tier == KnowledgeTier.ACTIVE

    def is_retrievable_hypothesis(self) -> bool:
        return self.tier == KnowledgeTier.HYPOTHESIS

    def to_public_dict(self) -> dict:
        """Safe representation for public reports — no raw refs, no PHI."""
        return {
            "safe_record_id": self.safe_record_id,
            "session_id": self.session_id,
            "fact_type": self.fact_type,
            "specialty": self.specialty,
            "source_type": self.source_type.value if isinstance(self.source_type, SourceType) else self.source_type,
            "source_ref": self.source_ref,   # must already be hashed by caller
            "trust_level": self.trust_level.value if isinstance(self.trust_level, TrustLevel) else self.trust_level,
            "tier": self.tier.value if isinstance(self.tier, KnowledgeTier) else self.tier,
            "status": self.status.value if isinstance(self.status, RecordStatus) else self.status,
            "confidence": self.confidence,
            "ddi_status": self.ddi_status.value if isinstance(self.ddi_status, DDIStatus) else self.ddi_status,
            "requires_review": self.requires_review,
        }


# ---------------------------------------------------------------------------
# LedgerEvent
# ---------------------------------------------------------------------------


@dataclass
class LedgerEvent:
    event_id: str
    event_type: LedgerEventType
    record_id: str
    timestamp: str
    actor: str = "system"
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    safe_public_details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.event_type in _RESERVED_EVENT_TYPES:
            raise ValueError(
                f"Event type {self.event_type} is reserved and not active in CKA-B01. "
                "Implement the corresponding block before using it."
            )

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value if isinstance(self.event_type, LedgerEventType) else self.event_type,
            "record_id": self.record_id,
            "timestamp": self.timestamp,
            "actor": self.actor,
            "reason": self.reason,
            "safe_public_details": self.safe_public_details,
        }
