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


def seed_connector_profile(
    metrics: PipelineMetrics,
    *,
    connector: str,
    confidence: float,
    latency_ms: float,
    success_rate: float,
    attempts: int = 4,
) -> None:
    successes = max(0, int(round(attempts * success_rate)))
    failures = max(attempts - successes, 0)
    for _ in range(successes):
        metrics.record_connector_result(
            connector=connector,
            latency_ms=latency_ms,
            confidence=confidence,
            success=True,
        )
    for _ in range(failures):
        metrics.record_connector_result(
            connector=connector,
            latency_ms=0.0,
            confidence=0.0,
            success=False,
        )


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
    seed_connector_profile(metrics, connector="spacy", confidence=0.6, latency_ms=4, success_rate=1.0)
    seed_connector_profile(metrics, connector="phi3", confidence=0.7, latency_ms=20, success_rate=0.98)
    seed_connector_profile(metrics, connector="gemini", confidence=0.88, latency_ms=110, success_rate=0.95)
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
    seed_connector_profile(metrics, connector="spacy", confidence=0.6, latency_ms=4, success_rate=1.0)
    seed_connector_profile(metrics, connector="phi3", confidence=0.7, latency_ms=20, success_rate=0.98)
    seed_connector_profile(metrics, connector="gemini", confidence=0.88, latency_ms=110, success_rate=0.95)
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


def test_cost_based_routing_prefers_cheapest_eligible_connector(tmp_path: Path):
    metrics = PipelineMetrics()
    seed_connector_profile(metrics, connector="spacy", confidence=0.6, latency_ms=4, success_rate=1.0)
    seed_connector_profile(metrics, connector="phi3", confidence=0.82, latency_ms=20, success_rate=0.98)
    seed_connector_profile(metrics, connector="gemini", confidence=0.91, latency_ms=120, success_rate=0.95)
    pipeline, audit_path = make_pipeline(
        tmp_path=tmp_path,
        spacy_extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [],
            "confidence": 0.6,
            "latency_ms": 4,
            "notes": [],
        }),
        gemini_extractor=StaticExtractor({
            "extractor": "gemini",
            "actual_extractor": "gemini",
            "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
            "confidence": 0.9,
            "latency_ms": 120,
            "notes": [],
        }, specialty="epilepsy"),
        phi3_extractor=StaticExtractor({
            "extractor": "phi3",
            "actual_extractor": "phi3",
            "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
            "confidence": 0.83,
            "latency_ms": 20,
            "notes": [],
        }),
        metrics=metrics,
    )

    result = pipeline.process_text("x" * 4500, specialty="epilepsy")

    assert result.audit["extractor_route"] == "phi3"
    assert result.audit["extractor_actual"] == "phi3"
    extraction_events = [event for event in read_jsonl(audit_path) if event["stage"] == "extraction"]
    assert "selected=phi3" in extraction_events[-1]["routing_decision_reason"]


def test_confidence_based_override_prefers_gemini_when_phi3_is_too_weak(tmp_path: Path):
    metrics = PipelineMetrics()
    seed_connector_profile(metrics, connector="spacy", confidence=0.6, latency_ms=4, success_rate=1.0)
    seed_connector_profile(metrics, connector="phi3", confidence=0.7, latency_ms=18, success_rate=0.98)
    seed_connector_profile(metrics, connector="gemini", confidence=0.89, latency_ms=110, success_rate=0.95)
    pipeline, _ = make_pipeline(
        tmp_path=tmp_path,
        spacy_extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [],
            "confidence": 0.6,
            "latency_ms": 4,
            "notes": [],
        }),
        gemini_extractor=StaticExtractor({
            "extractor": "gemini",
            "actual_extractor": "gemini",
            "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
            "confidence": 0.88,
            "latency_ms": 110,
            "notes": [],
        }, specialty="epilepsy"),
        phi3_extractor=StaticExtractor({
            "extractor": "phi3",
            "actual_extractor": "phi3",
            "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
            "confidence": 0.72,
            "latency_ms": 18,
            "notes": [],
        }),
        metrics=metrics,
    )

    result = pipeline.process_text("x" * 4500, specialty="epilepsy")

    assert result.audit["extractor_route"] == "gemini"
    assert result.audit["extractor_actual"] == "gemini"


def test_latency_based_fallback_moves_off_slow_connector(tmp_path: Path):
    metrics = PipelineMetrics()
    seed_connector_profile(metrics, connector="spacy", confidence=0.82, latency_ms=4, success_rate=1.0, attempts=10)
    seed_connector_profile(metrics, connector="phi3", confidence=0.84, latency_ms=20, success_rate=0.98)
    pipeline, _ = make_pipeline(
        tmp_path=tmp_path,
        spacy_extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
            "confidence": 0.85,
            "latency_ms": 250,
            "notes": [],
        }),
        phi3_extractor=StaticExtractor({
            "extractor": "phi3",
            "actual_extractor": "phi3",
            "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
            "confidence": 0.84,
            "latency_ms": 20,
            "notes": [],
        }),
        metrics=metrics,
    )

    result = pipeline.process_text("Diagnosis: Epilepsy.", specialty="epilepsy")

    assert result.audit["extractor_route"] == "spacy"
    assert result.audit["extractor_actual"] == "phi3"
    assert result.audit["fallback_used"] is True
    assert any("router_degraded=latency_too_high" in note for note in result.extractor_result["notes"])


def test_reliability_based_avoidance_skips_unreliable_gemini(tmp_path: Path):
    metrics = PipelineMetrics()
    seed_connector_profile(metrics, connector="phi3", confidence=0.81, latency_ms=22, success_rate=0.96, attempts=25)
    seed_connector_profile(metrics, connector="gemini", confidence=0.92, latency_ms=100, success_rate=0.25, attempts=20)
    pipeline, audit_path = make_pipeline(
        tmp_path=tmp_path,
        spacy_extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [],
            "confidence": 0.5,
            "latency_ms": 4,
            "notes": [],
        }),
        gemini_extractor=StaticExtractor({
            "extractor": "gemini",
            "actual_extractor": "gemini",
            "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
            "confidence": 0.92,
            "latency_ms": 100,
            "notes": [],
        }, specialty="epilepsy"),
        phi3_extractor=StaticExtractor({
            "extractor": "phi3",
            "actual_extractor": "phi3",
            "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
            "confidence": 0.82,
            "latency_ms": 22,
            "notes": [],
        }),
        metrics=metrics,
    )

    result = pipeline.process_text("x" * 4500, specialty="epilepsy")

    assert result.audit["extractor_route"] == "phi3"
    assert result.audit["extractor_actual"] == "phi3"
    extraction_events = [event for event in read_jsonl(audit_path) if event["stage"] == "extraction"]
    assert "success_rate=0.960" in extraction_events[-1]["routing_decision_reason"]
