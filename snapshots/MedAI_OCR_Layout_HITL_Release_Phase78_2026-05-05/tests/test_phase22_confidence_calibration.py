from __future__ import annotations

import json
from pathlib import Path

from execution.confidence_calibration import calibrate_confidence, classify_confidence_band
from execution.logging import AuditLogger
from execution.pipeline import ExecutionPipeline
from execution.review_queue import read_review_queue
from monitoring.observability import build_phase22_metrics, write_phase22_outputs


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


def make_summary(tmp_path: Path) -> dict:
    return {
        "generated_at": "2026-04-25T01:30:00+00:00",
        "dataset_dir": "test_data\\final_batch_50",
        "documents_selected": 3,
        "documents_processed": 2,
        "written": 1,
        "queued_for_review": 1,
        "external_quota_blocked": 1,
        "hard_failures": 0,
        "determinism": {"mode": "deterministic_path", "seed": None, "ordering": "sorted_pdf_listing"},
        "documents": [
            {
                "document": "high.pdf",
                "status": "processed",
                "outcome": "written",
                "validation_status": "accepted",
                "extractor_route": "spacy",
                "extractor_actual": "spacy",
                "extractor": "spacy",
                "confidence": 0.9,
                "raw_confidence": 0.9,
                "calibrated_confidence": 0.9,
                "confidence_band": "high",
                "calibration_reason": "raw_confidence_retained",
                "route_mismatch_flag": False,
                "review_recommendation": "accept",
            },
            {
                "document": "review.pdf",
                "status": "processed",
                "outcome": "queued_for_review",
                "validation_status": "needs_review",
                "extractor_route": "gemini",
                "extractor_actual": "rules_based",
                "extractor": "rules_based",
                "confidence": 0.65,
                "raw_confidence": 0.65,
                "calibrated_confidence": 0.65,
                "confidence_band": "review",
                "calibration_reason": "raw_confidence_retained,requested_route_mismatch_observed",
                "route_mismatch_flag": True,
                "review_recommendation": "operator_review",
            },
            {
                "document": "quota.pdf",
                "status": "external_quota_blocked",
                "outcome": "external_quota_blocked",
                "validation_status": "skipped_external_quota",
            },
        ],
    }


def test_confidence_band_assignment_is_deterministic():
    first = calibrate_confidence(
        raw_confidence=0.85,
        extractor_route="spacy",
        extractor_actual="spacy",
        requested_extractor_route="spacy",
    )
    second = calibrate_confidence(
        raw_confidence=0.85,
        extractor_route="spacy",
        extractor_actual="spacy",
        requested_extractor_route="spacy",
    )

    assert first == second
    assert classify_confidence_band(0.49) == "reject"
    assert classify_confidence_band(0.5) == "review"
    assert classify_confidence_band(0.7) == "acceptable"
    assert classify_confidence_band(0.85) == "high"


def test_high_confidence_does_not_enter_review(tmp_path: Path):
    pipeline = make_pipeline(
        extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
            "confidence": 0.92,
            "latency_ms": 1,
            "notes": [],
        }),
        tmp_path=tmp_path,
    )

    result = pipeline.process_text("Diagnosis: Epilepsy.", specialty="epilepsy")

    assert result.outcome == "written"
    assert result.audit["confidence_band"] == "high"
    assert result.audit["review_recommendation"] == "accept"
    assert read_review_queue(tmp_path / "review_queue.jsonl") == []


def test_review_band_confidence_enters_review_audit_path(tmp_path: Path):
    pipeline = make_pipeline(
        extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "Migraine"}],
            "confidence": 0.6,
            "latency_ms": 1,
            "notes": [],
        }),
        tmp_path=tmp_path,
    )

    result = pipeline.process_text("Diagnosis: Migraine.", specialty="neurology")
    queued = read_review_queue(tmp_path / "review_queue.jsonl")

    assert result.outcome == "queued_for_review"
    assert result.validation_status == "needs_review"
    assert result.audit["confidence_band"] == "review"
    assert result.audit["review_recommendation"] == "operator_review"
    assert len(queued) == 1
    assert queued[0]["confidence_band"] == "review"
    assert queued[0]["review_recommendation"] == "operator_review"


def test_reject_band_confidence_is_not_written_as_accepted_output(tmp_path: Path):
    pipeline = make_pipeline(
        extractor=StaticExtractor({
            "extractor": "spacy",
            "actual_extractor": "spacy",
            "entities": [{"type": "diagnosis", "text": "Unclear diagnosis"}],
            "confidence": 0.4,
            "latency_ms": 1,
            "notes": [],
        }),
        tmp_path=tmp_path,
    )

    result = pipeline.process_text("Diagnosis: Unclear diagnosis.", specialty="general")

    assert result.outcome == "queued_for_review"
    assert result.validation_status == "rejected"
    assert result.written_count == 0
    assert result.audit["confidence_band"] == "reject"
    assert result.audit["review_recommendation"] == "reject_do_not_write"


def test_route_mismatch_affects_calibration_audit_visibility(tmp_path: Path):
    gemini_extractor = StaticExtractor({
        "extractor": "gemini",
        "actual_extractor": "rules_based",
        "entities": [{"type": "diagnosis", "text": "Epilepsy"}],
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

    assert result.validation_status == "accepted"
    assert result.audit["route_mismatch_flag"] is True
    assert result.audit["confidence_band"] == "high"
    assert result.audit["review_recommendation"] == "accept_with_route_audit"
    assert "requested_route_mismatch_observed" in result.audit["calibration_reason"]


def test_phase22_metrics_are_emitted_deterministically(tmp_path: Path):
    summary = make_summary(tmp_path)
    first = build_phase22_metrics(summary)
    second = build_phase22_metrics(summary)

    assert first == second
    assert first["confidence_band_counts"] == {"high": 1, "review": 1}
    assert first["review_recommendation_counts"] == {"accept": 1, "operator_review": 1}
    assert first["route_mismatch_count"] == 1


def test_phase22_outputs_are_written(tmp_path: Path):
    artifact_path = tmp_path / "artifacts" / "phase22" / "confidence_calibration.json"
    report_path = tmp_path / "reports" / "phase22" / "accuracy_calibration_report.md"

    metrics = write_phase22_outputs(make_summary(tmp_path), artifact_path=artifact_path, report_path=report_path)

    assert artifact_path.exists()
    assert report_path.exists()
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == metrics
    assert "Phase 22 Accuracy Calibration Report" in report_path.read_text(encoding="utf-8")
