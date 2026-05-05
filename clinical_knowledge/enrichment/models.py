"""Data models for CKA-B06 Controlled Enrichment + Hypothesis Tier.

No real medical data. No real patient data. No real external connectors.
Synthetic structured data only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EnrichmentSourceKind(str, Enum):
    AI_RESPONSE = "ai_response"
    WEB_UNVERIFIED = "web_unverified"
    CONNECTOR_STUB = "connector_stub"
    MANUAL_REVIEW_PREPARED = "manual_review_prepared"


class EnrichmentCandidateStatus(str, Enum):
    CANDIDATE = "candidate"
    DUPLICATE_DISCARDED = "duplicate_discarded"
    QUEUED_PENDING_SAFETY = "queued_pending_safety"
    BLOCKED_SAFETY = "blocked_safety"
    WRITTEN_HYPOTHESIS = "written_hypothesis"
    CONFLICT_QUARANTINED = "conflict_quarantined"
    PROMOTION_PREPARED = "promotion_prepared"
    PROMOTION_BLOCKED = "promotion_blocked"


class EnrichmentAction(str, Enum):
    DISCARD_DUPLICATE = "discard_duplicate"
    WRITE_HYPOTHESIS = "write_hypothesis"
    QUEUE_PENDING_SAFETY = "queue_pending_safety"
    BLOCK_SAFETY = "block_safety"
    ROUTE_TRUTH_RESOLUTION = "route_truth_resolution"
    PREPARE_MANUAL_PROMOTION = "prepare_manual_promotion"
    BLOCK_AUTO_PROMOTION = "block_auto_promotion"


@dataclass
class EnrichmentCandidate:
    candidate_id: str
    safe_candidate_id: str           # hashed — never raw internal ID in public
    source_kind: EnrichmentSourceKind
    source_name: str                 # generic stub name — no real connector names
    source_response_hash: str        # SHA-256 hash of response — never raw response
    fact_type: str
    entity_text: str
    structured: Dict[str, Any] = field(default_factory=dict)
    specialty: str = "general"
    confidence: float = 0.0
    proposed_trust_level: int = 3    # 3=operator_reviewed for AI/connector; 4/5 for web
    proposed_tier: str = "hypothesis"  # always hypothesis for enrichment candidates
    extraction_method: str = "synthetic_structured_enrichment"
    safe_public_summary: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.proposed_tier != "hypothesis":
            raise ValueError("EnrichmentCandidate.proposed_tier must always be 'hypothesis'")
        if self.extraction_method != "synthetic_structured_enrichment":
            raise ValueError(
                "EnrichmentCandidate.extraction_method must be 'synthetic_structured_enrichment'"
            )


@dataclass
class EnrichmentQueueItem:
    queue_id: str
    safe_queue_id: str
    candidate_safe_id: str
    reason: str   # pending_ddi_check | blocked_high_ddi | pending_medium_ack |
                  # truth_resolution_review | safe_mode_enrichment_disabled | auto_promotion_blocked
    status: str   # pending | resolved | discarded
    created_at: str
    safe_public_summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnrichmentWriteResult:
    action: EnrichmentAction
    status: EnrichmentCandidateStatus
    explanation: str
    ledger_event_ready: bool
    written_record: Optional[Any] = None          # MKBRecord if written
    queued_item: Optional[EnrichmentQueueItem] = None
    truth_resolution_result: Optional[Any] = None  # TruthResolutionResult if used
    medication_gate_result: Optional[Any] = None   # MedicationWriteGateResult if used
    safe_public_summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromotionDecision:
    candidate_record_id: str
    allowed: bool
    auto_promotion_attempted: bool
    promotion_mode: str    # "auto_blocked" | "manual_prepared" | "blocked_pending_safety"
    reason: str
    requires_manual_review: bool
    ledger_event_ready: bool
    safe_public_summary: Dict[str, Any] = field(default_factory=dict)
