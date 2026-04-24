from __future__ import annotations

from dataclasses import dataclass, field

from app.config import ENABLE_TRUTH_RESOLUTION, TIER_QUARANTINED, TIER_SUPERSEDED
from app.schemas import MKBRecord
from execution.truth_resolution import ResolutionBatch, ResolutionDecision, TruthResolutionResolver
from governance.governance_ledger import GovernanceLedger


@dataclass(frozen=True)
class GovernanceResolution:
    action: str
    confidence: float
    winner: MKBRecord | None = None
    reason: str = ""
    requires_review: bool = False
    existing_id: str | None = None
    quarantined_records: list[MKBRecord] = field(default_factory=list)


class GovernanceTruthResolutionEngine:
    """Phase 11 governance ruleset. Default disabled through adapter."""

    def resolve(self, candidate: MKBRecord, existing: MKBRecord) -> GovernanceResolution:
        if self._is_materially_identical(candidate, existing):
            merged = existing.model_copy(update={
                "content": candidate.content or existing.content,
                "structured": {**existing.structured, **candidate.structured},
                "confidence": max(existing.confidence, candidate.confidence),
                "resolution_action": "merge",
                "resolution_confidence": 0.95,
                "requires_review": False,
            })
            return GovernanceResolution(
                action="merge",
                confidence=0.95,
                winner=merged,
                existing_id=existing.id,
                reason="materially_identical_records_merge",
            )

        if existing.trust_level == 1 and candidate.trust_level > 1:
            return GovernanceResolution(
                action="keep_existing",
                confidence=0.95,
                winner=existing,
                existing_id=existing.id,
                reason="trust_level_1_beats_lower_trust",
            )

        if candidate.trust_level == 1 and existing.trust_level > 1:
            return GovernanceResolution(
                action="replace_with_new",
                confidence=0.95,
                winner=candidate,
                existing_id=existing.id,
                reason="trust_level_1_beats_lower_trust",
            )

        if existing.trust_level == 2 and candidate.trust_level >= 3:
            return GovernanceResolution(
                action="keep_existing",
                confidence=0.9,
                winner=existing,
                existing_id=existing.id,
                reason="peer_review_beats_ai",
            )

        if candidate.trust_level == 2 and existing.trust_level >= 3:
            return GovernanceResolution(
                action="replace_with_new",
                confidence=0.9,
                winner=candidate,
                existing_id=existing.id,
                reason="peer_review_beats_ai",
            )

        date_gap = abs((candidate.first_recorded - existing.first_recorded).days)
        if candidate.trust_level == existing.trust_level and date_gap > 90:
            winner = candidate if candidate.first_recorded > existing.first_recorded else existing
            action = "replace_with_new" if winner is candidate else "keep_existing"
            return GovernanceResolution(
                action=action,
                confidence=0.8,
                winner=winner,
                existing_id=existing.id,
                reason="same_trust_newer_than_90_days",
            )

        if candidate.source_count >= 2 and existing.source_count <= 1 and candidate.trust_level <= existing.trust_level:
            return GovernanceResolution(
                action="replace_with_new",
                confidence=0.78,
                winner=candidate,
                existing_id=existing.id,
                reason="multi_source_replaces_weaker_single_source",
            )

        if self._can_merge_numeric_range(candidate, existing):
            merged = self._merge_numeric_range(candidate, existing)
            return GovernanceResolution(
                action="merge",
                confidence=0.72,
                winner=merged,
                existing_id=existing.id,
                reason="numeric_same_period_merge",
            )

        if self._is_medication_dose_conflict(candidate, existing):
            quarantined_candidate = self._quarantine(candidate, reason="medication_dose_conflict")
            quarantined_existing = self._quarantine(existing, reason="medication_dose_conflict")
            return GovernanceResolution(
                action="quarantine",
                confidence=0.0,
                winner=None,
                existing_id=existing.id,
                reason="medication_dose_conflict",
                requires_review=True,
                quarantined_records=[quarantined_existing, quarantined_candidate],
            )

        return GovernanceResolution(
            action="quarantine",
            confidence=0.0,
            winner=None,
            existing_id=existing.id,
            reason="unresolved_conflict_quarantines_candidate",
            requires_review=True,
            quarantined_records=[self._quarantine(candidate, reason="unresolved_conflict")],
        )

    def _can_merge_numeric_range(self, candidate: MKBRecord, existing: MKBRecord) -> bool:
        if self._numeric_value(candidate) is None or self._numeric_value(existing) is None:
            return False
        if self._entity_key(candidate) != self._entity_key(existing):
            return False
        return abs((candidate.first_recorded - existing.first_recorded).days) <= 90

    def _merge_numeric_range(self, candidate: MKBRecord, existing: MKBRecord) -> MKBRecord:
        values = [self._numeric_value(candidate), self._numeric_value(existing)]
        min_value = min(value for value in values if value is not None)
        max_value = max(value for value in values if value is not None)
        merged_structured = {
            **existing.structured,
            **candidate.structured,
            "range_min": min_value,
            "range_max": max_value,
            "merged_from": [existing.id, candidate.id],
        }
        return existing.model_copy(update={
            "content": f"{existing.content} [{min_value}-{max_value}]",
            "structured": merged_structured,
            "confidence": max(existing.confidence, candidate.confidence),
            "resolution_action": "merge",
            "resolution_confidence": 0.72,
        })

    def _is_medication_dose_conflict(self, candidate: MKBRecord, existing: MKBRecord) -> bool:
        return (
            candidate.fact_type == "medication"
            and existing.fact_type == "medication"
            and self._entity_key(candidate) == self._entity_key(existing)
            and (candidate.structured.get("dose") or "") != (existing.structured.get("dose") or "")
        )

    def _numeric_value(self, record: MKBRecord) -> float | None:
        value = record.structured.get("value")
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _is_materially_identical(self, candidate: MKBRecord, existing: MKBRecord) -> bool:
        if self._entity_key(candidate) != self._entity_key(existing):
            return False
        if candidate.fact_type != existing.fact_type:
            return False
        if candidate.trust_level != existing.trust_level:
            return False
        if candidate.source_count != existing.source_count:
            return False
        if abs((candidate.first_recorded - existing.first_recorded).days) > 90:
            return False
        if self._normalize_structured(candidate.structured) != self._normalize_structured(existing.structured):
            return False
        if self._numeric_value(candidate) is not None or self._numeric_value(existing) is not None:
            return self._numeric_value(candidate) == self._numeric_value(existing)
        return True

    def _normalize_structured(self, structured: dict) -> dict:
        return {
            str(key): value
            for key, value in structured.items()
            if key not in {"merged_from", "range_min", "range_max"}
        }

    def _entity_key(self, record: MKBRecord) -> tuple[str, str]:
        label = record.structured.get("name") or record.structured.get("text") or record.content
        return record.fact_type, str(label).strip().lower()

    def _quarantine(self, record: MKBRecord, *, reason: str) -> MKBRecord:
        return record.model_copy(update={
            "tier": TIER_QUARANTINED,
            "status": "quarantined",
            "requires_review": True,
            "resolution_action": "quarantine",
            "resolution_confidence": 0.0,
            "tags": list(dict.fromkeys([*record.tags, reason])),
        })


class GovernanceTruthResolutionAdapter:
    """Minimal adapter hook for the pipeline. Preserves Phase 10 behavior when disabled."""

    def __init__(
        self,
        existing_records_provider=None,
        *,
        enabled: bool = ENABLE_TRUTH_RESOLUTION,
        ledger: GovernanceLedger | None = None,
    ):
        self.enabled = enabled
        self.existing_records_provider = existing_records_provider or (lambda record: [])
        self.phase10_resolver = TruthResolutionResolver(self.existing_records_provider)
        self.engine = GovernanceTruthResolutionEngine()
        self.ledger = ledger or GovernanceLedger()

    def resolve_batch(self, records: list[MKBRecord]) -> ResolutionBatch:
        if not self.enabled:
            return self.phase10_resolver.resolve_batch(records)

        records_to_write: list[MKBRecord] = []
        quarantined_records: list[MKBRecord] = []
        decisions: list[ResolutionDecision] = []
        simulated_existing: list[MKBRecord] = []

        for record in records:
            existing = self._find_same_entity(record, list(self.existing_records_provider(record)) + simulated_existing)
            if existing is None:
                annotated = self._annotate(record, action="replace_with_new", confidence=1.0)
                records_to_write.append(annotated)
                simulated_existing.append(annotated)
                decisions.append(ResolutionDecision(
                    action="replace_with_new",
                    confidence=1.0,
                    record=annotated,
                    reason="No matching existing entity found.",
                ))
                self.ledger.log(event_type="truth_resolution", record_id=annotated.id, action="replace_with_new", details={"reason": "no_match"})
                continue

            resolution = self.engine.resolve(record, existing)
            self.ledger.log(
                event_type="truth_resolution",
                record_id=record.id,
                action=resolution.action,
                details={"reason": resolution.reason, "existing_id": resolution.existing_id},
            )

            if resolution.action == "keep_existing":
                decisions.append(ResolutionDecision(
                    action="keep_existing",
                    confidence=resolution.confidence,
                    record=existing,
                    reason=resolution.reason,
                    existing_id=existing.id,
                ))
                continue

            if resolution.action == "replace_with_new" and resolution.winner is not None:
                replacement = self._annotate(
                    resolution.winner.model_copy(update={"id": existing.id}),
                    action="replace_with_new",
                    confidence=resolution.confidence,
                )
                records_to_write = [item for item in records_to_write if item.id != existing.id]
                simulated_existing = [item for item in simulated_existing if item.id != existing.id]
                records_to_write.append(replacement)
                simulated_existing.append(replacement)
                decisions.append(ResolutionDecision(
                    action="replace_with_new",
                    confidence=resolution.confidence,
                    record=replacement,
                    reason=resolution.reason,
                    existing_id=existing.id,
                ))
                continue

            if resolution.action == "merge" and resolution.winner is not None:
                merged = self._annotate(resolution.winner, action="merge", confidence=resolution.confidence)
                records_to_write = [item for item in records_to_write if item.id != existing.id]
                simulated_existing = [item for item in simulated_existing if item.id != existing.id]
                records_to_write.append(merged)
                simulated_existing.append(merged)
                decisions.append(ResolutionDecision(
                    action="merge",
                    confidence=resolution.confidence,
                    record=merged,
                    reason=resolution.reason,
                    existing_id=existing.id,
                ))
                continue

            for quarantined in resolution.quarantined_records:
                quarantined_records.append(quarantined)
                decisions.append(ResolutionDecision(
                    action="quarantine",
                    confidence=resolution.confidence,
                    record=quarantined,
                    requires_review=True,
                    reason=resolution.reason,
                    existing_id=existing.id,
                ))

        return ResolutionBatch(
            records_to_write=records_to_write,
            quarantined_records=quarantined_records,
            decisions=decisions,
        )

    def _find_same_entity(self, candidate: MKBRecord, existing_records: list[MKBRecord]) -> MKBRecord | None:
        candidate_key = self.engine._entity_key(candidate)
        for existing in reversed(existing_records):
            if self.engine._entity_key(existing) == candidate_key:
                return existing
        return None

    def _annotate(self, record: MKBRecord, *, action: str, confidence: float) -> MKBRecord:
        return record.model_copy(update={
            "resolution_action": action,
            "resolution_confidence": confidence,
            "tier": record.tier if action != "replace_with_new" else record.tier,
            "status": record.status if action != "replace_with_new" else record.status,
        })
