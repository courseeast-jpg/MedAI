from __future__ import annotations

import json
from pathlib import Path

from execution.logging import AuditLogger
from execution.metrics import PipelineMetrics
from execution.pipeline import ExecutionPipeline
from execution.routing_efficiency import build_routing_efficiency, estimate_route_cost
from execution.review_queue import ReviewQueueWriter, read_review_queue
from monitoring.observability import build_phase23_metrics, write_phase23_outputs


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


def make_pipeline(
    *,
    extractor,
    tmp_path: Path,
    gemini_extractor=None,
    phi3_extractor=None,
    metrics: PipelineMetrics | None = None,
) -> ExecutionPipeline:
    return ExecutionPipeline(
        pii_stripper=NoopPIIStripper(),
        spacy_extractor=extractor,
        gemini_extractor=gemini_extractor,
        phi3_extractor=phi3_extractor,
        audit_logger=AuditLogger(path=tmp_path / "audit.jsonl"),
        review_queue_path=tmp_path / "review_queue.jsonl",
        pipeline_metrics=metrics or PipelineMetrics(),
    )


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


def make_summary(tmp_path: Path) -> dict:
    review_queue_path = tmp_path / "review_queue.jsonl"
    ReviewQueueWriter(review_queue_path).append_validation_review(
        run_id="run-1",
        document_id="review.pdf",
        source_filename="review.pdf",
        reason="confidence_below_accept_threshold",
        reasons=["confidence_below_accept_threshold"],
        confidence=0.68,
        extractor_route="phi3",
        extractor_actual="phi3",
        recommended_action="operator_review_validation",
        raw_evidence_path="review.pdf",
        requested_extractor_route="gemini",
        validation_status="needs_review",
        validation_errors=[],
        entity_count=2,
        notes=[],
        raw_confidence=0.68,
        calibrated_confidence=0.68,
        confidence_band="review",
        calibration_reason="raw_confidence_retained",
        route_mismatch_flag=True,
        review_recommendation="operator_review",
        intended_route="gemini",
        actual_route="phi3",
        fallback_reason="quota exceeded",
        estimated_cost_units=0.005,
        saved_cost_units=0.015,
        quota_block_avoided=True,
    )

    return {
        "generated_at": "2026-04-25T03:00:00+00:00",
        "dataset_dir": "test_data\\final_batch_50",
        "documents_selected": 3,
        "documents_processed": 2,
        "written": 1,
        "queued_for_review": 1,
        "external_quota_blocked": 1,
        "hard_failures": 0,
        "determinism": {"mode": "deterministic_path", "seed": None, "ordering": "sorted_pdf_listing"},
        "review_queue": {"path": str(review_queue_path), "items": 1},
        "documents": [
            {
                "document": "written.pdf",
                "status": "processed",
                "outcome": "written",
                "validation_status": "accepted",
                "intended_route": "spacy",
                "actual_route": "spacy",
                "extractor_route": "spacy",
                "extractor_actual": "spacy",
                "confidence_band": "acceptable",
                "review_recommendation": "accept",
                "route_mismatch_flag": False,
                "fallback_reason": None,
                "estimated_cost_units": 0.0,
                "saved_cost_units": 0.0,
                "quota_block_avoided": False,
            },
            {
                "document": "review.pdf",
                "status": "processed",
                "outcome": "queued_for_review",
                "validation_status": "needs_review",
                "intended_route": "gemini",
                "actual_route": "phi3",
                "extractor_route": "phi3",
                "extractor_actual": "phi3",
                "confidence_band": "review",
                "review_recommendation": "operator_review",
                "route_mismatch_flag": True,
                "fallback_reason": "quota exceeded",
                "estimated_cost_units": 0.005,
                "saved_cost_units": 0.015,
                "quota_block_avoided": True,
            },
            {
                "document": "quota.pdf",
                "status": "external_quota_blocked",
                "outcome": "external_quota_blocked",
                "validation_status": "skipped_external_quota",
            },
        ],
    }


def test_cost_units_are_deterministic():
    first = build_routing_efficiency(
        intended_route="gemini",
        actual_route="phi3",
        fallback_reason="quota exceeded",
        quota_block_avoided=True,
        confidence_band="review",
        review_recommendation="operator_review",
    )
    second = build_routing_efficiency(
        intended_route="gemini",
        actual_route="phi3",
        fallback_reason="quota exceeded",
        quota_block_avoided=True,
        confidence_band="review",
        review_recommendation="operator_review",
    )

    assert first == second
    assert estimate_route_cost("spacy") == 0.0
    assert estimate_route_cost("phi3") == 0.005
    assert estimate_route_cost("gemini") == 0.02


def test_saved_cost_units_are_calculated():
    efficiency = build_routing_efficiency(
        intended_route="gemini",
        actual_route="phi3",
        fallback_reason="quota exceeded",
        quota_block_avoided=True,
        confidence_band="review",
        review_recommendation="operator_review",
    )

    assert efficiency.estimated_cost_units == 0.005
    assert efficiency.saved_cost_units == 0.015
    assert efficiency.route_mismatch_flag is True


def test_route_mismatch_is_visible_not_hidden(tmp_path: Path):
    metrics = build_phase23_metrics(make_summary(tmp_path))

    assert metrics["route_mismatch_count"] == 1
    assert metrics["documents"][1]["intended_route"] == "gemini"
    assert metrics["documents"][1]["actual_route"] == "phi3"
    assert metrics["documents"][1]["route_mismatch_flag"] is True


def test_quota_safe_blocks_are_counted_separately_from_hard_failures(tmp_path: Path):
    metrics = build_phase23_metrics(make_summary(tmp_path))

    assert metrics["external_quota_blocked"] == 1
    assert metrics["hard_failures"] == 0
    assert metrics["quota_block_avoided_count"] == 1


def test_review_band_document_remains_review_audit_visible(tmp_path: Path):
    metrics = build_phase23_metrics(make_summary(tmp_path))
    queued = read_review_queue(tmp_path / "review_queue.jsonl")

    assert metrics["confidence_band_counts"] == {"acceptable": 1, "review": 1}
    assert metrics["review_recommendation_counts"] == {"accept": 1, "operator_review": 1}
    assert queued[0]["confidence_band"] == "review"
    assert queued[0]["review_recommendation"] == "operator_review"


def test_quota_blocked_gemini_reroutes_future_noisy_documents_to_phi3(tmp_path: Path):
    metrics = PipelineMetrics()
    seed_connector_profile(metrics, connector="spacy", confidence=0.6, latency_ms=4, success_rate=1.0)
    seed_connector_profile(metrics, connector="phi3", confidence=0.7, latency_ms=20, success_rate=0.98)
    seed_connector_profile(metrics, connector="gemini", confidence=0.9, latency_ms=110, success_rate=0.95)
    pipeline = make_pipeline(
        extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [],
            "confidence": 0.6,
            "latency_ms": 1,
            "notes": [],
        }),
        gemini_extractor=RaisingExtractor(RuntimeError("429 quota exceeded; retry in 12 seconds"), specialty="epilepsy"),
        phi3_extractor=StaticExtractor({
            "extractor": "phi3",
            "actual_extractor": "phi3",
            "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
            "confidence": 0.82,
            "latency_ms": 1,
            "notes": [],
        }),
        tmp_path=tmp_path,
        metrics=metrics,
    )

    first = pipeline.process_text("x" * 4500, specialty="epilepsy")
    second = pipeline.process_text("y" * 4500, specialty="epilepsy")

    assert first.audit["extractor_actual"] == "phi3"
    assert first.audit["intended_route"] == "gemini"
    assert second.audit["intended_route"] == "phi3"
    assert second.audit["actual_route"] == "phi3"
    assert second.audit["quota_block_avoided"] is True


def test_long_noisy_03_preserves_phi3_review_path_after_gemini_fallback(tmp_path: Path):
    metrics = PipelineMetrics()
    seed_connector_profile(metrics, connector="spacy", confidence=0.9, latency_ms=5, success_rate=0.99)
    seed_connector_profile(metrics, connector="phi3", confidence=0.8, latency_ms=20, success_rate=0.98)
    seed_connector_profile(metrics, connector="gemini", confidence=0.9, latency_ms=110, success_rate=0.95)
    pipeline = make_pipeline(
        extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
            "confidence": 0.7,
            "latency_ms": 1,
            "notes": [],
        }),
        gemini_extractor=RaisingExtractor(RuntimeError("429 quota exceeded; retry in 12 seconds"), specialty="epilepsy"),
        phi3_extractor=StaticExtractor({
            "extractor": "phi3",
            "actual_extractor": "phi3",
            "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
            "confidence": 0.68,
            "latency_ms": 1,
            "notes": [],
        }),
        tmp_path=tmp_path,
        metrics=metrics,
    )

    repeated_sentence = (
        "Hospital medication review. Diagnosis: seizure disorder. Levetiracetam 1000mg twice daily. "
        "Valproate 500mg nightly. EEG showed abnormal temporal sharp waves. Sodium 139 mmol/L. "
        "Recommendation: continue neurology follow up and monitor fatigue. "
    )
    result = pipeline.process_text(repeated_sentence * 7, specialty="epilepsy", source_name="long_noisy_03.pdf")
    queued = read_review_queue(tmp_path / "review_queue.jsonl")

    assert result.audit["intended_route"] == "gemini"
    assert result.audit["actual_route"] == "phi3"
    assert result.audit["raw_confidence"] == 0.68
    assert result.audit["calibrated_confidence"] == 0.68
    assert result.audit["confidence_band"] == "review"
    assert result.outcome == "queued_for_review"
    assert queued[0]["review_recommendation"] == "operator_review"


def test_long_noisy_03_stays_on_phi3_review_path_even_if_gemini_is_available(tmp_path: Path):
    metrics = PipelineMetrics()
    seed_connector_profile(metrics, connector="spacy", confidence=0.9, latency_ms=5, success_rate=0.99)
    seed_connector_profile(metrics, connector="phi3", confidence=0.8, latency_ms=20, success_rate=0.98)
    seed_connector_profile(metrics, connector="gemini", confidence=0.9, latency_ms=110, success_rate=0.95)
    pipeline = make_pipeline(
        extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
            "confidence": 0.7,
            "latency_ms": 1,
            "notes": [],
        }),
        gemini_extractor=StaticExtractor({
            "extractor": "gemini",
            "actual_extractor": "gemini",
            "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
            "confidence": 0.9,
            "latency_ms": 1,
            "notes": [],
        }, specialty="epilepsy"),
        phi3_extractor=StaticExtractor({
            "extractor": "phi3",
            "actual_extractor": "phi3",
            "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
            "confidence": 0.68,
            "latency_ms": 1,
            "notes": [],
        }),
        tmp_path=tmp_path,
        metrics=metrics,
    )

    repeated_sentence = (
        "Hospital medication review. Diagnosis: seizure disorder. Levetiracetam 1000mg twice daily. "
        "Valproate 500mg nightly. EEG showed abnormal temporal sharp waves. Sodium 139 mmol/L. "
        "Recommendation: continue neurology follow up and monitor fatigue. "
    )
    result = pipeline.process_text(repeated_sentence * 7, specialty="epilepsy", source_name="long_noisy_03.pdf")
    queued = read_review_queue(tmp_path / "review_queue.jsonl")

    assert result.audit["intended_route"] == "gemini"
    assert result.audit["actual_route"] == "phi3"
    assert result.audit["confidence_band"] == "review"
    assert result.outcome == "queued_for_review"
    assert any("router_fallback=gemini:preserve_review_route" in note for note in result.notes)
    assert queued[0]["review_recommendation"] == "operator_review"


def test_phase23_outputs_are_written(tmp_path: Path):
    artifact_path = tmp_path / "artifacts" / "phase23" / "routing_efficiency.json"
    report_path = tmp_path / "reports" / "phase23" / "routing_efficiency_report.md"

    metrics = write_phase23_outputs(make_summary(tmp_path), artifact_path=artifact_path, report_path=report_path)

    assert artifact_path.exists()
    assert report_path.exists()
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == metrics
    assert "Phase 23 Routing Efficiency Report" in report_path.read_text(encoding="utf-8")


def test_phase23_metrics_normalize_missing_fallback_reasons(tmp_path: Path):
    summary = make_summary(tmp_path)
    summary["documents"][0]["fallback_reason"] = "None"

    metrics = build_phase23_metrics(summary)

    assert metrics["fallback_reason_counts"] == {"quota exceeded": 1}
    assert metrics["documents"][0]["fallback_reason"] is None
