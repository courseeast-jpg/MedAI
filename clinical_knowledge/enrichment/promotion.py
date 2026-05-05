"""Hypothesis promotion logic for CKA-B06 Controlled Enrichment.

ENRICH_PROMOTE defaults to False — auto-promotion is blocked.

Rules:
- ENRICH_PROMOTE=False blocks ALL auto-promotion.
- manual_review_confirmed=True prepares a promotion decision but does NOT
  automatically rewrite active facts in this block.
- No medication records with unresolved/pending/high DDI may be promoted.
- No quarantined/conflicted records may be promoted.
- No promotion based on AI confidence alone.
"""
from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

from clinical_knowledge.enrichment.models import (
    EnrichmentAction,
    PromotionDecision,
)

if TYPE_CHECKING:
    from clinical_knowledge.config import CKAConfig
    from clinical_knowledge.store import MKBStore

_DDI_BLOCKING_STATUSES = {"high_blocked", "blocked", "pending", "unavailable"}


def prepare_hypothesis_promotion(
    record: Any,
    config: "CKAConfig",
    *,
    manual_review_confirmed: bool = False,
    store: Optional["MKBStore"] = None,
) -> PromotionDecision:
    """Prepare (but never auto-execute) a promotion decision.

    Returns PromotionDecision with allowed=True only when:
      - ENRICH_PROMOTE is True (not default), AND
      - manual_review_confirmed is True

    Even when allowed=True, this block does NOT rewrite the record to ACTIVE.
    The promotion_mode is set to "manual_prepared" only.
    """
    safe_id = getattr(record, "safe_record_id", "") or ""
    record_id = getattr(record, "record_id", "") or ""
    tier = getattr(record, "tier", None)
    ddi_status = getattr(record, "ddi_status", None)

    # Reject quarantined records
    from clinical_knowledge.models import KnowledgeTier
    if tier is not None:
        tier_val = tier.value if hasattr(tier, "value") else str(tier)
        if tier_val in ("quarantined", "superseded"):
            return PromotionDecision(
                candidate_record_id=record_id,
                allowed=False,
                auto_promotion_attempted=False,
                promotion_mode="blocked_quarantined",
                reason="Record is quarantined or superseded — promotion not allowed.",
                requires_manual_review=True,
                ledger_event_ready=False,
                safe_public_summary={
                    "safe_record_id": safe_id,
                    "allowed": False,
                    "promotion_mode": "blocked_quarantined",
                    "synthetic": True,
                },
            )

    # Reject records with blocking DDI status
    if ddi_status is not None:
        ddi_val = ddi_status.value if hasattr(ddi_status, "value") else str(ddi_status)
        if ddi_val in _DDI_BLOCKING_STATUSES:
            return PromotionDecision(
                candidate_record_id=record_id,
                allowed=False,
                auto_promotion_attempted=False,
                promotion_mode="blocked_pending_safety",
                reason=f"Record has unresolved DDI status ({ddi_val}) — promotion blocked.",
                requires_manual_review=True,
                ledger_event_ready=False,
                safe_public_summary={
                    "safe_record_id": safe_id,
                    "allowed": False,
                    "promotion_mode": "blocked_pending_safety",
                    "ddi_status": ddi_val,
                    "synthetic": True,
                },
            )

    enrich_promote = getattr(config, "ENRICH_PROMOTE", False)

    if not enrich_promote:
        return PromotionDecision(
            candidate_record_id=record_id,
            allowed=False,
            auto_promotion_attempted=False,
            promotion_mode="auto_blocked",
            reason="ENRICH_PROMOTE=False — auto-promotion is disabled.",
            requires_manual_review=True,
            ledger_event_ready=False,
            safe_public_summary={
                "safe_record_id": safe_id,
                "allowed": False,
                "promotion_mode": "auto_blocked",
                "enrich_promote": False,
                "synthetic": True,
            },
        )

    # ENRICH_PROMOTE=True path (non-default)
    if not manual_review_confirmed:
        return PromotionDecision(
            candidate_record_id=record_id,
            allowed=False,
            auto_promotion_attempted=True,
            promotion_mode="auto_blocked",
            reason="Auto-promotion requires manual_review_confirmed=True.",
            requires_manual_review=True,
            ledger_event_ready=False,
            safe_public_summary={
                "safe_record_id": safe_id,
                "allowed": False,
                "promotion_mode": "auto_blocked",
                "enrich_promote": True,
                "manual_review_confirmed": False,
                "synthetic": True,
            },
        )

    # Manual review confirmed — prepare only, do not auto-execute
    return PromotionDecision(
        candidate_record_id=record_id,
        allowed=True,
        auto_promotion_attempted=False,
        promotion_mode="manual_prepared",
        reason="Manual review confirmed — promotion decision prepared (not auto-executed).",
        requires_manual_review=True,
        ledger_event_ready=True,
        safe_public_summary={
            "safe_record_id": safe_id,
            "allowed": True,
            "promotion_mode": "manual_prepared",
            "enrich_promote": True,
            "manual_review_confirmed": True,
            "synthetic": True,
        },
    )
