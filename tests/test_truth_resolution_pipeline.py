from __future__ import annotations

import json
from pathlib import Path

from app.schemas import MKBRecord
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


def make_record(*, name: str, fact_type: str = "diagnosis", confidence: float = 0.9, structured: dict | None = None):
    base_structured = {"name": name}
    if structured:
        base_structured.update(structured)
    return MKBRecord(
        fact_type=fact_type,
        content=f"{fact_type.title()}: {name}",
        structured=base_structured,
        specialty="epilepsy",
        source_type="document",
        source_name="existing.txt",
        trust_level=1,
        confidence=confidence,
    )


def make_pipeline(*, payload: dict, existing_records: list[MKBRecord], tmp_path: Path) -> ExecutionPipeline:
    review_queue_path = tmp_path / "review_queue.jsonl"
    return ExecutionPipeline(
        pii_stripper=NoopPIIStripper(),
        spacy_extractor=StaticExtractor(payload),
        existing_records_provider=lambda record: existing_records,
        audit_logger=AuditLogger(path=tmp_path / "audit.jsonl"),
        review_queue_path=review_queue_path,
    )


def read_queue(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_identical_entities_merge_without_review(tmp_path: Path):
    existing = make_record(name="Epilepsy", structured={"status": "active"})
    pipeline = make_pipeline(
        payload={
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "Epilepsy", "structured": {"status": "active"}}],
            "confidence": 0.9,
            "latency_ms": 1,
            "notes": [],
        },
        existing_records=[existing],
        tmp_path=tmp_path,
    )

    result = pipeline.process_text("Diagnosis: Epilepsy.", specialty="epilepsy")

    assert result.outcome == "written"
    assert result.queued_count == 0
    assert result.written_count == 1
    assert result.records[0].resolution_action == "merge"
    assert result.records[0].resolution_confidence == 0.95
    assert result.records[0].requires_review is False


def test_conflicting_entities_replace_with_new_without_review(tmp_path: Path):
    existing = make_record(name="Epilepsy", confidence=0.6, structured={"status": "inactive"})
    pipeline = make_pipeline(
        payload={
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "Epilepsy", "structured": {"status": "active"}}],
            "confidence": 0.9,
            "latency_ms": 1,
            "notes": [],
        },
        existing_records=[existing],
        tmp_path=tmp_path,
    )

    result = pipeline.process_text("Diagnosis: Epilepsy.", specialty="epilepsy")

    assert result.outcome == "written"
    assert result.queued_count == 0
    assert result.written_count == 1
    assert result.records[0].resolution_action == "replace_with_new"
    assert result.records[0].resolution_confidence == 0.8


def test_keep_existing_conflict_does_not_enter_review(tmp_path: Path):
    existing = make_record(name="Epilepsy", confidence=0.95, structured={"status": "inactive"})
    pipeline = make_pipeline(
        payload={
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "Epilepsy", "structured": {"status": "active"}}],
            "confidence": 0.7,
            "latency_ms": 1,
            "notes": [],
        },
        existing_records=[existing],
        tmp_path=tmp_path,
    )

    result = pipeline.process_text("Diagnosis: Epilepsy.", specialty="epilepsy")

    assert result.outcome == "written"
    assert result.written_count == 0
    assert result.queued_count == 0
    assert read_queue(tmp_path / "review_queue.jsonl") == []


def test_quarantine_case_enters_review_queue(tmp_path: Path):
    existing = make_record(name="Epilepsy", confidence=0.8, structured={"status": "inactive"})
    pipeline = make_pipeline(
        payload={
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "Epilepsy", "structured": {"status": "active"}}],
            "confidence": 0.8,
            "latency_ms": 1,
            "notes": [],
        },
        existing_records=[existing],
        tmp_path=tmp_path,
    )

    result = pipeline.process_text("Diagnosis: Epilepsy.", specialty="epilepsy")

    assert result.outcome == "queued_for_review"
    assert result.written_count == 0
    assert result.queued_count == 1
    assert result.queued_records[0].resolution_action == "quarantine"
    assert result.queued_records[0].resolution_confidence == 0.0
    assert result.queued_records[0].requires_review is True
    queued = read_queue(tmp_path / "review_queue.jsonl")
    assert len(queued) == 1
    assert queued[0]["resolution_action"] == "quarantine"
