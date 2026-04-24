from __future__ import annotations

from pathlib import Path

from app.schemas import MKBRecord
from execution.enrichment import ControlledEnrichment
from execution.logging import AuditLogger
from execution.medication_safety_gate import MedicationSafetyGate
from execution.pipeline import ExecutionPipeline
from execution.promotion import HypothesisPromotion
from execution.truth_resolution import TruthResolutionResolver


class NoopPIIStripper:
    def strip(self, text: str):
        return text, "noop"


class StaticExtractor:
    def __init__(self, payload: dict):
        self.payload = payload

    def extract(self, text: str) -> dict:
        return {**self.payload, "raw_text": text}


class DuplicateDiagnosisEnrichment:
    def enrich(self, records: list[MKBRecord]) -> list[MKBRecord]:
        if not records:
            return []
        base = records[0]
        return [MKBRecord(
            fact_type="diagnosis",
            content=base.content,
            structured=dict(base.structured),
            specialty=base.specialty,
            source_type="enrichment",
            source_name="duplicate_enrichment",
            trust_level=3,
            confidence=0.88,
            enrichment_confidence=0.88,
            tier="hypothesis",
            requires_review=True,
            session_id=base.session_id,
        )]


class UniqueHypothesisEnrichment:
    def enrich(self, records: list[MKBRecord]) -> list[MKBRecord]:
        if not records:
            return []
        base = records[0]
        return [MKBRecord(
            fact_type="note",
            content="Hypothesis relation: additional review",
            structured={"text": "additional review", "derived_from": base.id},
            specialty=base.specialty,
            source_type="enrichment",
            source_name="unique_enrichment",
            trust_level=3,
            confidence=0.7,
            enrichment_confidence=0.7,
            tier="hypothesis",
            requires_review=True,
            session_id=base.session_id,
        )]


def make_record(*, fact_type: str, name: str, confidence: float = 0.9, structured: dict | None = None):
    payload = {"name": name} if fact_type in {"diagnosis", "medication"} else {"text": name}
    if structured:
        payload.update(structured)
    return MKBRecord(
        fact_type=fact_type,
        content=f"{fact_type.title()}: {name}",
        structured=payload,
        specialty="epilepsy",
        source_type="enrichment",
        source_name="test",
        trust_level=3,
        confidence=confidence,
        tier="hypothesis",
        requires_review=True,
    )


def build_pipeline(*, payload: dict, enrichment_engine, tmp_path: Path, existing_records_provider=None) -> ExecutionPipeline:
    provider = existing_records_provider or (lambda record: [])
    return ExecutionPipeline(
        pii_stripper=NoopPIIStripper(),
        spacy_extractor=StaticExtractor(payload),
        enrichment_engine=enrichment_engine,
        existing_records_provider=provider,
        audit_logger=AuditLogger(path=tmp_path / "audit.jsonl"),
        review_queue_path=tmp_path / "review_queue.jsonl",
    )


def test_promotion_when_corroborated(tmp_path: Path):
    pipeline = build_pipeline(
        payload={
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
            "confidence": 0.9,
            "latency_ms": 1,
            "notes": [],
        },
        enrichment_engine=DuplicateDiagnosisEnrichment(),
        tmp_path=tmp_path,
    )

    result = pipeline.process_text("Diagnosis: Epilepsy.", specialty="epilepsy")

    assert result.outcome == "written"
    promoted = [record for record in result.records if record.promotion_history]
    assert promoted
    assert promoted[0].tier == "active"
    assert promoted[0].trust_level <= 2
    assert promoted[0].source_count >= 2


def test_no_promotion_when_single_source(tmp_path: Path):
    pipeline = build_pipeline(
        payload={
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
            "confidence": 0.9,
            "latency_ms": 1,
            "notes": [],
        },
        enrichment_engine=UniqueHypothesisEnrichment(),
        tmp_path=tmp_path,
    )

    result = pipeline.process_text("Diagnosis: Epilepsy.", specialty="epilepsy")

    assert result.outcome == "queued_for_review"
    assert any(record.source_type == "enrichment" and record.tier == "hypothesis" for record in result.queued_records)
    assert all(not record.promotion_history for record in result.queued_records if record.source_type == "enrichment")


def test_promotion_respects_conflict_rules():
    hypothesis = make_record(fact_type="diagnosis", name="Epilepsy", confidence=0.9, structured={"status": "active"})
    extraction = hypothesis.model_copy(update={"source_type": "extraction", "tier": "active", "requires_review": False})
    existing = make_record(fact_type="diagnosis", name="Epilepsy", confidence=0.9, structured={"status": "inactive"})

    promoter = HypothesisPromotion(lambda record: [existing])
    batch = promoter.promote([hypothesis], corroborating_records=[extraction])
    assert len(batch.promoted_records) == 1

    resolver = TruthResolutionResolver(lambda record: [existing])
    resolution = resolver.resolve_batch(batch.promoted_records)
    assert resolution.quarantined_records
    assert resolution.quarantined_records[0].requires_review is True


def test_promotion_respects_safety_gate():
    hypothesis_med = make_record(fact_type="medication", name="Lamotrigine", confidence=0.9)
    extraction_med = hypothesis_med.model_copy(update={"source_type": "extraction", "tier": "active", "requires_review": False})

    promoter = HypothesisPromotion(lambda record: [])
    batch = promoter.promote([hypothesis_med], corroborating_records=[extraction_med])
    promoted = batch.promoted_records[0]

    gate = MedicationSafetyGate(active_medications_provider=lambda: [make_record(fact_type="medication", name="Valproate")])
    decision, _, findings = gate.gate_medication_write(promoted)
    assert decision == "block"
    assert promoted.ddi_status == "high_blocked"
    assert findings
