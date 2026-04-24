from __future__ import annotations

import json
from pathlib import Path

from execution.audit import StageAuditLogger
from execution.logging import AuditLogger
from execution.metrics import PipelineMetrics
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


class RaisingExtractor:
    def __init__(self, exc: Exception, *, specialty: str = "general"):
        self.exc = exc
        self.specialty = specialty

    def extract(self, text: str) -> dict:
        raise self.exc


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_pipeline(
    *,
    tmp_path: Path,
    spacy_extractor,
    gemini_extractor=None,
    phi3_extractor=None,
    metrics: PipelineMetrics | None = None,
) -> tuple[ExecutionPipeline, Path]:
    audit_path = tmp_path / "stages.jsonl"
    pipeline = ExecutionPipeline(
        pii_stripper=NoopPIIStripper(),
        spacy_extractor=spacy_extractor,
        gemini_extractor=gemini_extractor,
        phi3_extractor=phi3_extractor,
        audit_logger=AuditLogger(path=tmp_path / "audit.jsonl"),
        stage_audit_logger=StageAuditLogger(path=audit_path),
        pipeline_metrics=metrics or PipelineMetrics(),
        review_queue_path=tmp_path / "review_queue.jsonl",
    )
    return pipeline, audit_path


def test_normal_routing_uses_spacy_for_simple_text(tmp_path: Path):
    pipeline, _ = make_pipeline(
        tmp_path=tmp_path,
        spacy_extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
            "confidence": 0.9,
            "latency_ms": 1,
            "notes": [],
        }),
    )

    result = pipeline.process_text("Diagnosis: Epilepsy.", specialty="epilepsy")

    assert result.outcome == "written"
    assert result.audit["extractor_route"] == "spacy"
    assert result.audit["extractor_actual"] == "spacy"
    assert pipeline.metrics.snapshot()["spacy_count"] == 1
    assert pipeline.metrics.snapshot()["gemini_count"] == 0


def test_gemini_failure_falls_back_to_phi3(tmp_path: Path):
    metrics = PipelineMetrics()
    pipeline, audit_path = make_pipeline(
        tmp_path=tmp_path,
        spacy_extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
            "confidence": 0.7,
            "latency_ms": 1,
            "notes": [],
        }),
        gemini_extractor=RaisingExtractor(RuntimeError("gemini connector error"), specialty="epilepsy"),
        phi3_extractor=StaticExtractor({
            "extractor": "phi3",
            "actual_extractor": "phi3",
            "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
            "confidence": 0.82,
            "latency_ms": 1,
            "notes": [],
        }),
        metrics=metrics,
    )

    result = pipeline.process_text("x" * 4000, specialty="epilepsy")

    assert result.outcome == "queued_for_review"
    assert result.validation_status == "needs_review"
    assert result.audit["extractor_route"] == "gemini"
    assert result.audit["extractor_actual"] == "phi3"
    assert result.audit["fallback_used"] is True
    assert metrics.snapshot()["fallback_count"] == 1
    assert metrics.snapshot()["failure_count"] == 1
    extraction_events = [event for event in read_jsonl(audit_path) if event["stage"] == "extraction"]
    assert extraction_events[-1]["extractor_route"] == "gemini"
    assert extraction_events[-1]["extractor_actual"] == "phi3"


def test_timeout_handling_uses_fallback(tmp_path: Path):
    metrics = PipelineMetrics()
    pipeline, _ = make_pipeline(
        tmp_path=tmp_path,
        spacy_extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "Migraine"}],
            "confidence": 0.7,
            "latency_ms": 1,
            "notes": [],
        }),
        gemini_extractor=RaisingExtractor(TimeoutError("gemini timeout"), specialty="epilepsy"),
        phi3_extractor=StaticExtractor({
            "extractor": "phi3",
            "actual_extractor": "phi3",
            "entities": [{"type": "diagnosis", "text": "Migraine"}],
            "confidence": 0.84,
            "latency_ms": 1,
            "notes": [],
        }),
        metrics=metrics,
    )

    result = pipeline.process_text("x" * 5000, specialty="epilepsy")

    assert result.outcome == "queued_for_review"
    assert result.extractor_result["fallback_used"] is True
    assert any("router_fallback=gemini:timeout" in note for note in result.notes)
    assert metrics.snapshot()["failure_count"] == 1


def test_route_vs_actual_mismatch_is_audited(tmp_path: Path):
    pipeline, audit_path = make_pipeline(
        tmp_path=tmp_path,
        spacy_extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [],
            "confidence": 0.7,
            "latency_ms": 1,
            "notes": [],
        }),
        gemini_extractor=StaticExtractor({
            "extractor": "gemini",
            "actual_extractor": "rules_based",
            "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
            "confidence": 0.85,
            "latency_ms": 1,
            "notes": ["gemini_route_legacy_fallback=rules_based"],
        }, specialty="epilepsy"),
    )

    result = pipeline.process_text("x" * 4000, specialty="epilepsy")

    assert result.outcome == "queued_for_review"
    assert result.validation_status == "rejected"
    assert result.audit["extractor_route"] == "gemini"
    assert result.audit["extractor_actual"] == "rules_based"
    extraction_events = [event for event in read_jsonl(audit_path) if event["stage"] == "extraction"]
    assert any("route_actual_mismatch" in json.dumps(event) for event in extraction_events)
