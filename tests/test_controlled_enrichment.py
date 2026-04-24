from __future__ import annotations

import json
from pathlib import Path

from app.schemas import MKBRecord
from execution.enrichment import ControlledEnrichment
from execution.logging import AuditLogger
from execution.pipeline import ExecutionPipeline


class NoopPIIStripper:
    def strip(self, text: str):
        return text, "noop"


class StaticExtractor:
    def __init__(self, payload: dict):
        self.payload = payload

    def extract(self, text: str) -> dict:
        return {**self.payload, "raw_text": text}


def make_record(*, fact_type: str, name: str, specialty: str = "epilepsy") -> MKBRecord:
    structured = {"name": name} if fact_type in {"diagnosis", "medication"} else {"text": name}
    return MKBRecord(
        fact_type=fact_type,
        content=f"{fact_type.title()}: {name}",
        structured=structured,
        specialty=specialty,
        source_type="extraction",
        source_name="existing.txt",
        trust_level=1,
        confidence=0.9,
    )


def build_pipeline(
    *,
    payload: dict,
    tmp_path: Path,
    existing_records_provider=None,
    active_medications_provider=None,
) -> ExecutionPipeline:
    return ExecutionPipeline(
        pii_stripper=NoopPIIStripper(),
        spacy_extractor=StaticExtractor(payload),
        existing_records_provider=existing_records_provider,
        active_medications_provider=active_medications_provider,
        enrichment_engine=ControlledEnrichment(existing_records_provider),
        audit_logger=AuditLogger(path=tmp_path / "audit.jsonl"),
        review_queue_path=tmp_path / "review_queue.jsonl",
    )


def read_queue(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_enrichment_creates_new_records(tmp_path: Path):
    pipeline = build_pipeline(
        payload={
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
            "confidence": 0.9,
            "latency_ms": 1,
            "notes": [],
        },
        tmp_path=tmp_path,
        existing_records_provider=lambda record: [],
    )

    result = pipeline.process_text("Diagnosis: Epilepsy.", specialty="epilepsy")

    assert result.outcome == "queued_for_review"
    assert result.written_count == 1
    assert any(record.source_type == "enrichment" for record in result.queued_records)


def test_enrichment_never_overwrites_existing(tmp_path: Path):
    existing_recommendation = make_record(fact_type="recommendation", name="review management plan for Epilepsy")
    pipeline = build_pipeline(
        payload={
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
            "confidence": 0.9,
            "latency_ms": 1,
            "notes": [],
        },
        tmp_path=tmp_path,
        existing_records_provider=lambda record: [existing_recommendation] if record.fact_type == "recommendation" else [],
    )

    result = pipeline.process_text("Diagnosis: Epilepsy.", specialty="epilepsy")

    assert result.written_count == 1
    assert all(record.source_type != "enrichment" for record in result.queued_records)


def test_all_enrichment_outputs_are_hypothesis(tmp_path: Path):
    pipeline = build_pipeline(
        payload={
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "test_result", "text": "EEG", "structured": {"value": "abnormal"}}],
            "confidence": 0.9,
            "latency_ms": 1,
            "notes": [],
        },
        tmp_path=tmp_path,
        existing_records_provider=lambda record: [],
    )

    result = pipeline.process_text("EEG abnormal.", specialty="epilepsy")

    enrichment_records = [record for record in result.queued_records if record.source_type == "enrichment"]
    assert enrichment_records
    assert all(record.tier == "hypothesis" for record in enrichment_records)
    assert all(record.requires_review is True for record in enrichment_records)
    assert all(record.enrichment_confidence is not None for record in enrichment_records)


def test_enrichment_medication_respects_safety_gate(tmp_path: Path):
    pipeline = build_pipeline(
        payload={
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "recommendation", "text": "Consider medication: Lamotrigine"}],
            "confidence": 0.9,
            "latency_ms": 1,
            "notes": [],
        },
        tmp_path=tmp_path,
        existing_records_provider=lambda record: [],
        active_medications_provider=lambda: [make_record(fact_type="medication", name="Valproate")],
    )

    result = pipeline.process_text("Consider medication: Lamotrigine", specialty="epilepsy")

    assert result.written_count == 1
    enrichment_medications = [record for record in result.queued_records if record.source_type == "enrichment" and record.fact_type == "medication"]
    assert enrichment_medications
    assert enrichment_medications[0].ddi_status == "high_blocked"
    queue_items = read_queue(tmp_path / "review_queue.jsonl")
    assert any(item.get("ddi_status") == "high_blocked" for item in queue_items)
