"""Deterministic Phase 3 truth resolution for execution candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from app.config import TIER_QUARANTINED
from app.schemas import MKBRecord


@dataclass(frozen=True)
class ResolutionDecision:
    action: str
    confidence: float
    record: MKBRecord | None = None
    requires_review: bool = False
    reason: str = ""
    existing_id: str | None = None


@dataclass(frozen=True)
class ResolutionBatch:
    records_to_write: list[MKBRecord] = field(default_factory=list)
    quarantined_records: list[MKBRecord] = field(default_factory=list)
    decisions: list[ResolutionDecision] = field(default_factory=list)


class TruthResolutionResolver:
    """Resolves incoming extraction records against existing records deterministically."""

    def __init__(self, existing_records_provider: Callable[[MKBRecord], list[MKBRecord]] | None = None):
        self.existing_records_provider = existing_records_provider or (lambda record: [])

    def resolve_batch(self, records: list[MKBRecord]) -> ResolutionBatch:
        records_to_write: list[MKBRecord] = []
        quarantined_records: list[MKBRecord] = []
        decisions: list[ResolutionDecision] = []
        simulated_existing: list[MKBRecord] = []

        for record in records:
            existing_records = list(self.existing_records_provider(record)) + simulated_existing
            matching = self._find_same_entity(record, existing_records)
            if matching is None:
                resolved = self._annotate(record, "replace_with_new", 1.0)
                records_to_write.append(resolved)
                simulated_existing.append(resolved)
                decisions.append(ResolutionDecision(
                    action="replace_with_new",
                    confidence=1.0,
                    record=resolved,
                    reason="No matching existing entity found.",
                ))
                continue

            if self._value_signature(record) == self._value_signature(matching):
                merged = self._merge_records(record, matching)
                records_to_write = [item for item in records_to_write if item.id != matching.id]
                records_to_write.append(merged)
                simulated_existing = [item for item in simulated_existing if item.id != matching.id]
                simulated_existing.append(merged)
                decisions.append(ResolutionDecision(
                    action="merge",
                    confidence=0.95,
                    record=merged,
                    reason="Same entity and same value merged deterministically.",
                    existing_id=matching.id,
                ))
                continue

            confidence_delta = round(record.confidence - matching.confidence, 3)
            if abs(confidence_delta) >= 0.1:
                if confidence_delta > 0:
                    replacement = self._annotate(
                        record.model_copy(update={"id": matching.id}),
                        "replace_with_new",
                        0.8,
                    )
                    records_to_write = [item for item in records_to_write if item.id != matching.id]
                    records_to_write.append(replacement)
                    simulated_existing = [item for item in simulated_existing if item.id != matching.id]
                    simulated_existing.append(replacement)
                    decisions.append(ResolutionDecision(
                        action="replace_with_new",
                        confidence=0.8,
                        record=replacement,
                        reason="Same entity conflict resolved in favor of higher-confidence new record.",
                        existing_id=matching.id,
                    ))
                else:
                    kept = self._annotate(matching, "keep_existing", 0.8)
                    simulated_existing = [item for item in simulated_existing if item.id != matching.id]
                    simulated_existing.append(kept)
                    decisions.append(ResolutionDecision(
                        action="keep_existing",
                        confidence=0.8,
                        record=kept,
                        reason="Same entity conflict resolved in favor of higher-confidence existing record.",
                        existing_id=matching.id,
                    ))
                continue

            quarantined = self._annotate(
                record.model_copy(update={
                    "tier": TIER_QUARANTINED,
                    "status": "quarantined",
                    "requires_review": True,
                }),
                "quarantine",
                0.0,
                requires_review=True,
            )
            quarantined_records.append(quarantined)
            decisions.append(ResolutionDecision(
                action="quarantine",
                confidence=0.0,
                record=quarantined,
                requires_review=True,
                reason="Same entity conflict could not be resolved deterministically.",
                existing_id=matching.id,
            ))

        return ResolutionBatch(
            records_to_write=records_to_write,
            quarantined_records=quarantined_records,
            decisions=decisions,
        )

    def _find_same_entity(self, candidate: MKBRecord, existing_records: list[MKBRecord]) -> MKBRecord | None:
        candidate_key = self._entity_key(candidate)
        for existing in reversed(existing_records):
            if self._entity_key(existing) == candidate_key:
                return existing
        return None

    def _entity_key(self, record: MKBRecord) -> tuple[str, str]:
        name = (
            record.structured.get("name")
            or record.structured.get("text")
            or record.content
        )
        return record.fact_type, str(name).strip().lower()

    def _value_signature(self, record: MKBRecord) -> tuple:
        structured = dict(record.structured or {})
        for key in ("name", "text", "validation_status", "validation_errors", "review_reasons"):
            structured.pop(key, None)
        return tuple(sorted((key, str(value)) for key, value in structured.items()))

    def _merge_records(self, candidate: MKBRecord, existing: MKBRecord) -> MKBRecord:
        merged_linked = list(dict.fromkeys([*existing.linked_to, existing.id, candidate.id]))
        merged_structured = {**existing.structured, **candidate.structured}
        merged_structured["merged_from"] = [existing.id, candidate.id]
        return self._annotate(existing.model_copy(update={
            "content": candidate.content if len(candidate.content) >= len(existing.content) else existing.content,
            "structured": merged_structured,
            "confidence": max(existing.confidence, candidate.confidence),
            "linked_to": merged_linked,
        }), "merge", 0.95)

    def _annotate(
        self,
        record: MKBRecord,
        action: str,
        confidence: float,
        *,
        requires_review: bool = False,
    ) -> MKBRecord:
        return record.model_copy(update={
            "resolution_action": action,
            "resolution_confidence": confidence,
            "requires_review": requires_review,
        })
