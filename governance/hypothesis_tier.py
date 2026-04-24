from __future__ import annotations

from app.config import ENABLE_HYPOTHESIS_TIER, TIER_ACTIVE, TIER_HYPOTHESIS, TRUST_AI
from app.schemas import MKBRecord


AI_EXTRACTION_METHODS = {"gemini", "claude", "phi3"}
AI_SOURCE_TYPES = {"ai_response", "web"}


class GovernanceHypothesisTier:
    """Classifies AI- or web-derived facts into hypothesis tier when enabled."""

    def __init__(self, *, enabled: bool = ENABLE_HYPOTHESIS_TIER, promotion_enabled: bool = False):
        self.enabled = enabled
        self.promotion_enabled = promotion_enabled

    def classify_record(self, record: MKBRecord) -> MKBRecord:
        if not self.enabled or not self._is_hypothesis_candidate(record):
            return record
        tags = list(dict.fromkeys([*record.tags, "hypothesis", "governance"]))
        return record.model_copy(update={
            "tier": TIER_HYPOTHESIS,
            "status": "hypothesis",
            "requires_review": True,
            "trust_level": max(record.trust_level, TRUST_AI),
            "tags": tags,
        })

    def active_context(self, records: list[MKBRecord]) -> list[MKBRecord]:
        return [
            record for record in records
            if record.tier == TIER_ACTIVE and record.status == "active"
        ]

    def manual_promote_placeholder(self, record: MKBRecord) -> MKBRecord:
        if not self.promotion_enabled:
            return record
        return record.model_copy(update={
            "tier": TIER_ACTIVE,
            "status": "active",
            "requires_review": False,
        })

    def _is_hypothesis_candidate(self, record: MKBRecord) -> bool:
        return (
            record.source_type in AI_SOURCE_TYPES
            or record.trust_level >= TRUST_AI
            or record.extraction_method in AI_EXTRACTION_METHODS
        )

