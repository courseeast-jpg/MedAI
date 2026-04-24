"""Deterministic hypothesis promotion and trust model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from app.config import TIER_ACTIVE, TIER_HYPOTHESIS
from app.schemas import MKBRecord


@dataclass(frozen=True)
class PromotionBatch:
    promoted_records: list[MKBRecord] = field(default_factory=list)
    remaining_hypotheses: list[MKBRecord] = field(default_factory=list)


class HypothesisPromotion:
    """Promotes corroborated hypothesis records into active tier."""

    def __init__(self, existing_records_provider: Callable[[MKBRecord], list[MKBRecord]] | None = None):
        self.existing_records_provider = existing_records_provider or (lambda record: [])

    def promote(self, records: list[MKBRecord], *, corroborating_records: list[MKBRecord]) -> PromotionBatch:
        promoted: list[MKBRecord] = []
        remaining: list[MKBRecord] = []

        for record in records:
            source_count = self._source_count(record, corroborating_records)
            promoted_record = record.model_copy(update={"source_count": source_count})
            should_promote = False
            reasons: list[str] = []

            if source_count >= 2:
                should_promote = True
                reasons.append("source_count>=2")
            if self._confirmed_by_extraction_and_enrichment(record, corroborating_records):
                should_promote = True
                reasons.append("confirmed_by_extraction_and_enrichment")
            if should_promote and promoted_record.confidence >= 0.85 and not self._has_conflict(record):
                reasons.append("high_confidence_no_conflict")

            if should_promote:
                promoted.append(self._promote_record(promoted_record, reasons))
            else:
                remaining.append(promoted_record.model_copy(update={"tier": TIER_HYPOTHESIS}))

        return PromotionBatch(promoted_records=promoted, remaining_hypotheses=remaining)

    def _source_count(self, record: MKBRecord, corroborating_records: list[MKBRecord]) -> int:
        count = max(int(record.source_count or 1), 1)
        corroborated = sum(1 for candidate in corroborating_records if self._record_key(candidate) == self._record_key(record))
        return count + corroborated

    def _confirmed_by_extraction_and_enrichment(self, record: MKBRecord, corroborating_records: list[MKBRecord]) -> bool:
        if record.source_type != "enrichment":
            return False
        return any(candidate.source_type == "extraction" and self._record_key(candidate) == self._record_key(record) for candidate in corroborating_records)

    def _has_conflict(self, record: MKBRecord) -> bool:
        for existing in self.existing_records_provider(record):
            if self._entity_key(existing) != self._entity_key(record):
                continue
            if self._value_signature(existing) != self._value_signature(record):
                return True
        return False

    def _promote_record(self, record: MKBRecord, reasons: list[str]) -> MKBRecord:
        history_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "from_tier": record.tier,
            "to_tier": TIER_ACTIVE,
            "reasons": reasons,
        }
        trust_level = 2 if record.source_count >= 2 else min(record.trust_level, 3)
        return record.model_copy(update={
            "tier": TIER_ACTIVE,
            "requires_review": False,
            "trust_level": trust_level,
            "promotion_history": [*record.promotion_history, history_entry],
        })

    def _record_key(self, record: MKBRecord) -> tuple[str, str]:
        label = record.structured.get("name") or record.structured.get("text") or record.content
        return record.fact_type, str(label).strip().lower()

    def _entity_key(self, record: MKBRecord) -> tuple[str, str]:
        base = record.structured.get("name") or record.structured.get("text") or record.content
        return record.fact_type, str(base).strip().lower()

    def _value_signature(self, record: MKBRecord) -> tuple:
        structured = dict(record.structured or {})
        for key in ("name", "text", "derived_from"):
            structured.pop(key, None)
        return tuple(sorted((key, str(value)) for key, value in structured.items()))
