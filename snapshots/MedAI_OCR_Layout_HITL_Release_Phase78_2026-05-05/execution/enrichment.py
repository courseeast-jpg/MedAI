"""Deterministic controlled enrichment for hypothesis-tier records."""

from __future__ import annotations

from typing import Callable

from app.config import TIER_HYPOTHESIS
from app.schemas import MKBRecord


class ControlledEnrichment:
    """Produces deterministic hypothesis-tier candidates from safe records."""

    def __init__(self, existing_records_provider: Callable[[MKBRecord], list[MKBRecord]] | None = None):
        self.existing_records_provider = existing_records_provider or (lambda record: [])

    def enrich(self, records: list[MKBRecord]) -> list[MKBRecord]:
        enriched: list[MKBRecord] = []
        seen = {self._record_key(record) for record in records}

        for record in records:
            for candidate in self._derive_candidates(record):
                key = self._record_key(candidate)
                if key in seen or self._exists(candidate):
                    continue
                seen.add(key)
                enriched.append(candidate)

        return enriched

    def _derive_candidates(self, record: MKBRecord) -> list[MKBRecord]:
        if record.fact_type == "diagnosis":
            name = record.structured.get("name") or record.content
            return [self._hypothesis_record(
                base=record,
                fact_type="recommendation",
                content=f"Hypothesis recommendation: review management plan for {name}",
                structured={"text": f"review management plan for {name}", "derived_from": record.id},
                enrichment_confidence=round(record.confidence * 0.7, 3),
            )]

        if record.fact_type == "test_result":
            label = record.structured.get("text") or record.structured.get("name") or record.content
            return [self._hypothesis_record(
                base=record,
                fact_type="note",
                content=f"Hypothesis relation: trend {label}",
                structured={"text": f"trend {label}", "relation": "trend", "derived_from": record.id},
                enrichment_confidence=round(record.confidence * 0.65, 3),
            )]

        if record.fact_type == "recommendation":
            text = str(record.structured.get("text") or record.content)
            prefix = "consider medication:"
            if text.lower().startswith(prefix):
                med_name = text[len(prefix):].strip()
                if med_name:
                    return [self._hypothesis_record(
                        base=record,
                        fact_type="medication",
                        content=f"Hypothesis medication: {med_name}",
                        structured={"name": med_name, "derived_from": record.id},
                        enrichment_confidence=round(record.confidence * 0.6, 3),
                    )]
            return []

        if record.fact_type == "medication":
            name = record.structured.get("name") or record.content
            return [self._hypothesis_record(
                base=record,
                fact_type="note",
                content=f"Hypothesis relation: reconcile medication {name}",
                structured={"text": f"reconcile medication {name}", "relation": "medication_reconciliation", "derived_from": record.id},
                enrichment_confidence=round(record.confidence * 0.6, 3),
            )]

        return []

    def _hypothesis_record(
        self,
        *,
        base: MKBRecord,
        fact_type: str,
        content: str,
        structured: dict,
        enrichment_confidence: float,
    ) -> MKBRecord:
        tags = list(dict.fromkeys([*base.tags, "hypothesis", "enrichment", fact_type]))
        return MKBRecord(
            fact_type=fact_type,
            content=content,
            structured=structured,
            specialty=base.specialty,
            source_type="enrichment",
            source_name="controlled_enrichment",
            trust_level=base.trust_level,
            confidence=enrichment_confidence,
            enrichment_confidence=enrichment_confidence,
            tier=TIER_HYPOTHESIS,
            extraction_method=base.extraction_method,
            ddi_checked=False,
            session_id=base.session_id,
            requires_review=True,
            tags=tags,
        )

    def _exists(self, candidate: MKBRecord) -> bool:
        return any(self._record_key(existing) == self._record_key(candidate) for existing in self.existing_records_provider(candidate))

    def _record_key(self, record: MKBRecord) -> tuple[str, str]:
        name = (
            record.structured.get("name")
            or record.structured.get("text")
            or record.content
        )
        return record.fact_type, str(name).strip().lower()
