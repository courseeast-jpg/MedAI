from __future__ import annotations

import json
from pathlib import Path

from execution.logging import AuditLogger
from execution.pipeline import ExecutionPipeline


class NoopPIIStripper:
    def strip(self, text: str):
        return text, "noop"


class StaticExtractor:
    def __init__(self, payload: dict, *, specialty: str = "general"):
        self.payload = payload
        self.specialty = specialty

    def extract(self, text: str) -> dict:
        return {**self.payload, "raw_text": text}


def make_pipeline(*, extractor, tmp_path: Path, gemini_extractor=None) -> ExecutionPipeline:
    return ExecutionPipeline(
        pii_stripper=NoopPIIStripper(),
        spacy_extractor=extractor,
        gemini_extractor=gemini_extractor,
        audit_logger=AuditLogger(path=tmp_path / "audit.jsonl"),
        review_queue_path=tmp_path / "review_queue.jsonl",
    )


def read_queue(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_valid_spacy_extraction_is_accepted(tmp_path: Path):
    pipeline = make_pipeline(
        extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
            "confidence": 0.9,
            "latency_ms": 1,
            "notes": [],
        }),
        tmp_path=tmp_path,
    )

    result = pipeline.process_text("Diagnosis: Epilepsy.", specialty="epilepsy")

    assert result.outcome == "written"
    assert result.validation_status == "accepted"
    assert result.validation_errors == []
    assert result.written_count == 1
    assert read_queue(tmp_path / "review_queue.jsonl") == []


def test_valid_gemini_extraction_is_accepted(tmp_path: Path):
    gemini_extractor = StaticExtractor({
        "extractor": "gemini",
        "actual_extractor": "gemini",
        "entities": [{"type": "medication", "text": "Lamotrigine", "structured": {"name": "Lamotrigine"}}],
        "confidence": 0.85,
        "latency_ms": 1,
        "notes": [],
    }, specialty="epilepsy")
    pipeline = make_pipeline(
        extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [],
            "confidence": 0.9,
            "latency_ms": 1,
            "notes": [],
        }),
        gemini_extractor=gemini_extractor,
        tmp_path=tmp_path,
    )

    result = pipeline.process_text("x" * 4000, specialty="epilepsy")

    assert result.outcome == "written"
    assert result.validation_status == "accepted"
    assert result.audit["extractor_route"] == "gemini"
    assert result.audit["extractor_actual"] == "gemini"
    assert result.written_count == 1


def test_missing_required_fields_are_rejected(tmp_path: Path):
    pipeline = make_pipeline(
        extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis"}],
            "confidence": 0.9,
            "latency_ms": 1,
            "notes": [],
        }),
        tmp_path=tmp_path,
    )

    result = pipeline.process_text("Diagnosis: Epilepsy.", specialty="epilepsy")

    assert result.outcome == "queued_for_review"
    assert result.validation_status == "rejected"
    assert any(error["code"] == "missing_required_field" for error in result.validation_errors)
    queued = read_queue(tmp_path / "review_queue.jsonl")
    assert len(queued) == 1
    assert queued[0]["validation_status"] == "rejected"


def test_low_confidence_routes_to_needs_review(tmp_path: Path):
    pipeline = make_pipeline(
        extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
            "confidence": 0.6,
            "latency_ms": 1,
            "notes": [],
        }),
        tmp_path=tmp_path,
    )

    result = pipeline.process_text("Diagnosis: Epilepsy.", specialty="epilepsy")

    assert result.outcome == "queued_for_review"
    assert result.validation_status == "needs_review"
    assert result.queued_count == 1
    assert any(error["code"] == "confidence_below_accept_threshold" for error in result.validation_errors)
    queued = read_queue(tmp_path / "review_queue.jsonl")
    assert len(queued) == 1
    assert queued[0]["reasons"] == ["confidence_below_accept_threshold"]


def test_gemini_fallback_violation_is_rejected(tmp_path: Path):
    gemini_extractor = StaticExtractor({
        "extractor": "gemini",
        "actual_extractor": "rules_based",
        "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
        "confidence": 0.65,
        "latency_ms": 1,
        "notes": [],
    }, specialty="epilepsy")
    pipeline = make_pipeline(
        extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [],
            "confidence": 0.9,
            "latency_ms": 1,
            "notes": [],
        }),
        gemini_extractor=gemini_extractor,
        tmp_path=tmp_path,
    )

    result = pipeline.process_text("x" * 4000, specialty="epilepsy")

    assert result.outcome == "queued_for_review"
    assert result.validation_status == "rejected"
    assert any(error["code"] == "route_actual_mismatch" for error in result.validation_errors)


def test_malformed_extraction_payload_is_rejected(tmp_path: Path):
    pipeline = make_pipeline(
        extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "Epilepsy", "structured": "bad"}],
            "confidence": 0.9,
            "latency_ms": 1,
            "notes": [],
        }),
        tmp_path=tmp_path,
    )

    result = pipeline.process_text("Diagnosis: Epilepsy.", specialty="epilepsy")

    assert result.outcome == "queued_for_review"
    assert result.validation_status == "rejected"
    assert any(error["code"] == "invalid_schema_shape" for error in result.validation_errors)


def test_review_queue_and_metrics_capture_phase2_statuses(tmp_path: Path):
    review_queue_path = tmp_path / "review_queue.jsonl"
    audit_logger = AuditLogger(path=tmp_path / "audit.jsonl")

    accepted_pipeline = ExecutionPipeline(
        pii_stripper=NoopPIIStripper(),
        spacy_extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
            "confidence": 0.9,
            "latency_ms": 1,
            "notes": [],
        }),
        audit_logger=audit_logger,
        review_queue_path=review_queue_path,
    )
    review_pipeline = ExecutionPipeline(
        pii_stripper=NoopPIIStripper(),
        spacy_extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "Migraine"}],
            "confidence": 0.6,
            "latency_ms": 1,
            "notes": [],
        }),
        audit_logger=audit_logger,
        review_queue_path=review_queue_path,
    )
    rejected_pipeline = ExecutionPipeline(
        pii_stripper=NoopPIIStripper(),
        spacy_extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis"}],
            "confidence": 0.9,
            "latency_ms": 1,
            "notes": [],
        }),
        audit_logger=audit_logger,
        review_queue_path=review_queue_path,
    )

    accepted_pipeline.process_text("Diagnosis: Epilepsy.", specialty="epilepsy")
    review_pipeline.process_text("Diagnosis: Migraine.", specialty="epilepsy")
    rejected_pipeline.process_text("Diagnosis: Missing text.", specialty="epilepsy")

    queued = read_queue(review_queue_path)
    assert [item["validation_status"] for item in queued] == ["needs_review", "rejected"]
    assert all(item["reasons"] for item in queued)

    assert audit_logger.metrics.snapshot() == {
        "total_jobs": 3,
        "spacy_count": 3,
        "gemini_count": 0,
        "avg_confidence": 0.8,
        "review_rate": 0.667,
        "accepted_count": 1,
        "review_count": 1,
        "rejected_count": 1,
        "validation_error_count": 2,
        "avg_confidence_by_status": {
            "accepted": 0.9,
            "needs_review": 0.6,
            "rejected": 0.9,
        },
    }
