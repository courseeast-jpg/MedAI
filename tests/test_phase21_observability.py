from __future__ import annotations

import json
from pathlib import Path

from execution.review_queue import ReviewQueueWriter
from monitoring.observability import build_phase21_metrics, write_phase21_outputs


def make_summary(tmp_path: Path) -> dict:
    review_queue_path = tmp_path / "review_queue.jsonl"
    stage_audit_path = tmp_path / "pipeline_stages.jsonl"

    review_queue = ReviewQueueWriter(review_queue_path)
    review_queue.append_validation_review(
        run_id="run-1",
        document_id="doc-1.pdf",
        source_filename="doc-1.pdf",
        reason="confidence_below_accept_threshold",
        reasons=["confidence_below_accept_threshold"],
        confidence=0.68,
        extractor_route="phi3",
        extractor_actual="phi3",
        recommended_action="operator_review_validation",
        raw_evidence_path="doc-1.pdf",
        requested_extractor_route="gemini",
        validation_status="needs_review",
        validation_errors=[],
        entity_count=2,
        notes=[],
    )
    review_queue.append_external_quota_block(
        run_id="run-1",
        document_id="doc-3.pdf",
        source_filename="doc-3.pdf",
        reason="external_quota_blocked",
        recommended_action="operator_retry_after_quota_reset",
        raw_evidence_path="doc-3.pdf",
        error="429 quota exceeded; retry in 12 seconds",
        retry_visibility={"retry_detected": True, "retry_delay_seconds": 12.0, "retry_reason": "external_quota"},
    )

    stage_events = [
        {"timestamp": "2026-04-24T21:00:00.000+00:00", "record_id": "run-1-doc-1", "stage": "extraction"},
        {"timestamp": "2026-04-24T21:00:00.050+00:00", "record_id": "run-1-doc-1", "stage": "extraction"},
        {"timestamp": "2026-04-24T21:00:00.060+00:00", "record_id": "run-1-doc-1", "stage": "validation"},
        {"timestamp": "2026-04-24T21:00:00.000+00:00", "record_id": "run-1-doc-2", "stage": "extraction"},
        {"timestamp": "2026-04-24T21:00:00.040+00:00", "record_id": "run-1-doc-2", "stage": "extraction"},
    ]
    with stage_audit_path.open("w", encoding="utf-8") as handle:
        for event in stage_events:
            handle.write(json.dumps(event) + "\n")

    return {
        "generated_at": "2026-04-24T21:24:02.521094+00:00",
        "dataset_dir": "test_data\\final_batch_50",
        "documents_selected": 3,
        "documents_processed": 2,
        "written": 1,
        "queued_for_review": 1,
        "external_quota_blocked": 1,
        "hard_failures": 0,
        "review_queue": {"path": str(review_queue_path), "items": 2},
        "component_state": {
            "review_queue_path": str(review_queue_path),
            "stage_audit_log_path": str(stage_audit_path),
        },
        "aggregate": {
            "avg_confidence": 0.69,
        },
        "determinism": {"mode": "deterministic_path", "seed": None},
        "documents": [
            {
                "document": "doc-1.pdf",
                "status": "processed",
                "outcome": "queued_for_review",
                "extractor_route": "phi3",
                "extractor_actual": "phi3",
                "extractor": "phi3",
                "requested_route": "gemini",
                "confidence": 0.68,
                "processing_time_ms": 50.0,
            },
            {
                "document": "doc-2.pdf",
                "status": "processed",
                "outcome": "written",
                "extractor_route": "spacy",
                "extractor_actual": "spacy",
                "extractor": "spacy",
                "requested_route": "spacy",
                "confidence": 0.7,
                "processing_time_ms": 40.0,
            },
            {
                "document": "doc-3.pdf",
                "status": "external_quota_blocked",
                "outcome": "external_quota_blocked",
                "extractor_route": None,
                "extractor_actual": None,
                "extractor": None,
                "requested_route": None,
                "confidence": 0.0,
                "processing_time_ms": 12.0,
            },
        ],
    }


def test_phase21_metrics_are_emitted_deterministically(tmp_path: Path):
    summary = make_summary(tmp_path)

    first = build_phase21_metrics(summary)
    second = build_phase21_metrics(summary)

    assert first == second
    assert first["attempted_documents"] == 3
    assert first["processed_documents"] == 2
    assert first["written_documents"] == 1
    assert first["queued_for_review_documents"] == 1
    assert first["review_queue_items"] == 2
    assert first["external_quota_blocked"] == 1
    assert first["hard_failures"] == 0
    assert first["average_confidence"] == 0.69


def test_phase21_route_counts_and_review_queue_counts_are_captured(tmp_path: Path):
    metrics = build_phase21_metrics(make_summary(tmp_path))

    assert metrics["extractor_route_counts"] == {"phi3": 1, "spacy": 1}
    assert metrics["extractor_actual_counts"] == {"phi3": 1, "spacy": 1}
    assert metrics["route_mismatch_count"] == 1
    assert metrics["review_queue_items"] == 2
    assert metrics["review_queue_category_counts"] == {
        "external_quota_block": 1,
        "validation_review": 1,
    }


def test_phase21_quota_safe_blocks_are_counted_without_hard_failures(tmp_path: Path):
    metrics = build_phase21_metrics(make_summary(tmp_path))

    assert metrics["quota_safe_block_count"] == 1
    assert metrics["hard_failures"] == 0
    assert metrics["low_confidence_count"] == 0
    assert metrics["per_stage_duration_ms"]["extraction"]["total_duration_ms"] == 90.0


def test_phase21_outputs_are_written(tmp_path: Path):
    summary = make_summary(tmp_path)
    artifact_path = tmp_path / "artifacts" / "phase21" / "observability_metrics.json"
    report_path = tmp_path / "reports" / "phase21" / "observability_report.md"

    metrics = write_phase21_outputs(summary, artifact_path=artifact_path, report_path=report_path)

    assert artifact_path.exists()
    assert report_path.exists()
    written_metrics = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert written_metrics == metrics
    assert "Phase 21 Observability Report" in report_path.read_text(encoding="utf-8")
