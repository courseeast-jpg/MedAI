from __future__ import annotations

import json
from pathlib import Path

from scripts.run_phase12_real_world_validation import (
    build_phase15_aggregate,
    build_phase13_metrics,
    build_phase12_summary,
    write_phase15_reports,
    is_external_quota_error,
    summarize_document,
    write_phase13_reports,
)


def test_build_phase12_summary_aggregates_documents():
    documents = [
        {
            "document": "short_01.pdf",
            "status": "processed",
            "error": None,
            "outcome": "written",
            "validation_status": "accepted",
            "extractor": "spacy",
            "extractor_actual": "spacy",
            "confidence": 0.7,
            "entity_count": 2,
            "written_count": 2,
            "queued_count": 0,
            "blocked_count": 0,
            "review_reasons": [],
            "notes": [],
            "processing_time_ms": 12.5,
            "requested_route": "spacy",
            "retry_visibility": {"retry_detected": False, "retry_delay_seconds": None, "retry_reason": None},
        },
        {
            "document": "short_02.pdf",
            "status": "processed",
            "error": None,
            "outcome": "queued_for_review",
            "validation_status": "needs_review",
            "extractor": "gemini",
            "extractor_actual": "rules_based",
            "confidence": 0.45,
            "entity_count": 1,
            "written_count": 0,
            "queued_count": 1,
            "blocked_count": 0,
            "review_reasons": ["pending_validation_review"],
            "notes": [],
            "processing_time_ms": 30.0,
            "requested_route": "gemini",
            "retry_visibility": {"retry_detected": False, "retry_delay_seconds": None, "retry_reason": None},
        },
        {
            "document": "short_03.pdf",
            "status": "external_quota_blocked",
            "error": "429 quota exceeded for metric generate_content",
            "outcome": "external_quota_blocked",
            "validation_status": "skipped_external_quota",
            "extractor": None,
            "extractor_actual": None,
            "confidence": 0.0,
            "entity_count": 0,
            "written_count": 0,
            "queued_count": 0,
            "blocked_count": 0,
            "review_reasons": [],
            "notes": [],
            "processing_time_ms": 7.5,
            "requested_route": None,
            "retry_visibility": {"retry_detected": True, "retry_delay_seconds": None, "retry_reason": "external_quota"},
        },
        {
            "document": "short_04.pdf",
            "status": "error",
            "error": "boom",
            "outcome": "error",
            "validation_status": "error",
            "extractor": None,
            "extractor_actual": None,
            "confidence": 0.0,
            "entity_count": 0,
            "written_count": 0,
            "queued_count": 0,
            "blocked_count": 0,
            "review_reasons": [],
            "notes": [],
            "processing_time_ms": 4.0,
            "requested_route": None,
            "retry_visibility": {"retry_detected": False, "retry_delay_seconds": None, "retry_reason": None},
        },
    ]

    summary = build_phase12_summary(
        dataset_dir=Path("test_data/final_batch_50"),
        requested_limit=10,
        documents=documents,
        runtime_counts={"total": 3, "active": 2, "hypothesis": 0, "quarantined": 1},
        component_state={"governance_active": True},
    )

    assert summary["documents_processed"] == 2
    assert summary["written"] == 1
    assert summary["queued_for_review"] == 1
    assert summary["external_quota_blocked"] == 1
    assert summary["hard_failures"] == 1
    assert summary["documents_failed"] == 1
    assert summary["run_passed"] is False
    assert summary["aggregate"]["outcomes"] == {"queued_for_review": 1, "written": 1}
    assert summary["aggregate"]["validation_statuses"] == {"accepted": 1, "needs_review": 1}
    assert summary["aggregate"]["extractors"] == {"rules_based": 1, "spacy": 1}
    assert summary["aggregate"]["avg_confidence"] == 0.575
    assert summary["aggregate"]["total_entities"] == 3
    assert any("at least 10 documents" in item for item in summary["recommendations"])
    assert any("processing errors" in item for item in summary["recommendations"])
    assert any("quota exhaustion" in item for item in summary["recommendations"])
    assert summary["documents"][0]["processing_time_ms"] == 12.5


def test_quota_safe_mode_classifies_quota_exhaustion_as_external_block():
    document = summarize_document(
        Path("quota_blocked.pdf"),
        None,
        error=RuntimeError(
            "Gemini route fallback occurred despite configured key: rules_based; "
            "root_cause=429 You exceeded your current quota, please retry in 36 seconds."
        ),
        quota_safe=True,
    )

    assert is_external_quota_error(document["error"]) is True
    assert document["status"] == "external_quota_blocked"
    assert document["outcome"] == "external_quota_blocked"
    assert document["validation_status"] == "skipped_external_quota"
    assert "429" in document["error"]
    assert "route_actual_mismatch" not in document["review_reasons"]
    assert document["retry_visibility"]["retry_detected"] is True
    assert document["retry_visibility"]["retry_delay_seconds"] == 36.0


def test_quota_safe_mode_does_not_hide_normal_hard_failures():
    document = summarize_document(
        Path("hard_failure.pdf"),
        None,
        error=RuntimeError("corrupt PDF header"),
        quota_safe=True,
    )

    assert is_external_quota_error(document["error"]) is False
    assert document["status"] == "error"
    assert document["outcome"] == "error"
    assert document["validation_status"] == "error"
    assert document["retry_visibility"]["retry_detected"] is False


def test_phase13_reports_are_created_with_expected_counters(tmp_path: Path):
    documents = [
        {
            "document": "short_01.pdf",
            "status": "processed",
            "error": None,
            "outcome": "written",
            "validation_status": "accepted",
            "extractor": "spacy",
            "extractor_actual": "spacy",
            "confidence": 0.7,
            "entity_count": 2,
            "written_count": 2,
            "queued_count": 0,
            "blocked_count": 0,
            "review_reasons": [],
            "notes": ["routing_decision=selected=spacy score=0.90"],
            "processing_time_ms": 10.0,
            "requested_route": "spacy",
            "retry_visibility": {"retry_detected": False, "retry_delay_seconds": None, "retry_reason": None},
        },
        {
            "document": "short_02.pdf",
            "status": "processed",
            "error": None,
            "outcome": "queued_for_review",
            "validation_status": "needs_review",
            "extractor": "phi3",
            "extractor_actual": "phi3",
            "confidence": 0.5,
            "entity_count": 1,
            "written_count": 0,
            "queued_count": 1,
            "blocked_count": 0,
            "review_reasons": ["pending_validation_review"],
            "notes": ["routing_decision=selected=gemini score=0.70"],
            "processing_time_ms": 40.0,
            "requested_route": "gemini",
            "retry_visibility": {"retry_detected": False, "retry_delay_seconds": None, "retry_reason": None},
        },
        {
            "document": "short_03.pdf",
            "status": "external_quota_blocked",
            "error": "429 quota exceeded; please retry in 12.5s",
            "outcome": "external_quota_blocked",
            "validation_status": "skipped_external_quota",
            "extractor": None,
            "extractor_actual": None,
            "confidence": 0.0,
            "entity_count": 0,
            "written_count": 0,
            "queued_count": 0,
            "blocked_count": 0,
            "review_reasons": [],
            "notes": [],
            "processing_time_ms": 15.0,
            "requested_route": None,
            "retry_visibility": {"retry_detected": True, "retry_delay_seconds": 12.5, "retry_reason": "external_quota"},
        },
        {
            "document": "short_04.pdf",
            "status": "error",
            "error": "boom",
            "outcome": "error",
            "validation_status": "error",
            "extractor": None,
            "extractor_actual": None,
            "confidence": 0.0,
            "entity_count": 0,
            "written_count": 0,
            "queued_count": 0,
            "blocked_count": 0,
            "review_reasons": [],
            "notes": [],
            "processing_time_ms": 5.0,
            "requested_route": None,
            "retry_visibility": {"retry_detected": False, "retry_delay_seconds": None, "retry_reason": None},
        },
    ]

    summary = build_phase12_summary(
        dataset_dir=Path("test_data/final_batch_50"),
        requested_limit=10,
        documents=documents,
        runtime_counts={"total": 3, "active": 2, "hypothesis": 0, "quarantined": 1},
        component_state={"governance_active": True},
    )
    metrics = write_phase13_reports(tmp_path, summary)

    metrics_path = tmp_path / "metrics_snapshot.json"
    performance_path = tmp_path / "performance_summary.md"

    assert metrics_path.exists()
    assert performance_path.exists()

    written_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert written_metrics["documents_processed"] == 2
    assert written_metrics["written"] == 1
    assert written_metrics["queued_for_review"] == 1
    assert written_metrics["external_quota_blocked"] == 1
    assert written_metrics["hard_failures"] == 1
    assert written_metrics["avg_confidence"] == 0.6
    assert written_metrics["route_distribution"] == {"phi3": 1, "spacy": 1}
    assert written_metrics["requested_route_distribution"] == {"gemini": 1, "spacy": 1, "unknown": 2}
    assert written_metrics["timing"]["total_pipeline_time_ms"] == 70.0
    assert written_metrics["timing"]["per_extractor"]["spacy"]["avg_time_ms"] == 10.0
    assert written_metrics["retries"]["retry_event_count"] == 1
    assert metrics == written_metrics

    performance_summary = performance_path.read_text(encoding="utf-8")
    assert "Phase 13 Performance Summary" in performance_summary
    assert "Retry events observed: 1" in performance_summary
    assert "`short_03.pdf` -> status=external_quota_blocked" in performance_summary


def test_phase13_metrics_do_not_regress_phase12_counts():
    documents = [
        {
            "document": "short_01.pdf",
            "status": "processed",
            "error": None,
            "outcome": "written",
            "validation_status": "accepted",
            "extractor": "spacy",
            "extractor_actual": "spacy",
            "confidence": 0.7,
            "entity_count": 2,
            "written_count": 2,
            "queued_count": 0,
            "blocked_count": 0,
            "review_reasons": [],
            "notes": ["routing_decision=selected=spacy score=0.90"],
            "processing_time_ms": 10.0,
            "requested_route": "spacy",
            "retry_visibility": {"retry_detected": False, "retry_delay_seconds": None, "retry_reason": None},
        },
        {
            "document": "short_02.pdf",
            "status": "external_quota_blocked",
            "error": "429 quota exceeded; please retry in 8.0s",
            "outcome": "external_quota_blocked",
            "validation_status": "skipped_external_quota",
            "extractor": None,
            "extractor_actual": None,
            "confidence": 0.0,
            "entity_count": 0,
            "written_count": 0,
            "queued_count": 0,
            "blocked_count": 0,
            "review_reasons": [],
            "notes": [],
            "processing_time_ms": 8.0,
            "requested_route": None,
            "retry_visibility": {"retry_detected": True, "retry_delay_seconds": 8.0, "retry_reason": "external_quota"},
        },
    ]
    summary = build_phase12_summary(
        dataset_dir=Path("test_data/final_batch_50"),
        requested_limit=10,
        documents=documents,
        runtime_counts={"total": 2, "active": 1, "hypothesis": 0, "quarantined": 0},
        component_state={"governance_active": True},
    )
    metrics = build_phase13_metrics(summary)

    assert summary["written"] == 1
    assert summary["queued_for_review"] == 0
    assert summary["external_quota_blocked"] == 1
    assert summary["hard_failures"] == 0
    assert metrics["written"] == summary["written"]
    assert metrics["queued_for_review"] == summary["queued_for_review"]
    assert metrics["external_quota_blocked"] == summary["external_quota_blocked"]
    assert metrics["hard_failures"] == summary["hard_failures"]


def test_phase15_reports_are_created_with_expected_aggregate(tmp_path: Path):
    documents = [
        {
            "document": "short_01.pdf",
            "status": "processed",
            "error": None,
            "outcome": "written",
            "validation_status": "accepted",
            "extractor": "spacy",
            "extractor_actual": "spacy",
            "confidence": 0.7,
            "entity_count": 2,
            "written_count": 2,
            "queued_count": 0,
            "blocked_count": 0,
            "review_reasons": [],
            "notes": [],
            "processing_time_ms": 10.0,
            "requested_route": "spacy",
            "retry_visibility": {"retry_detected": False, "retry_delay_seconds": None, "retry_reason": None},
        },
        {
            "document": "short_02.pdf",
            "status": "processed",
            "error": None,
            "outcome": "queued_for_review",
            "validation_status": "needs_review",
            "extractor": "phi3",
            "extractor_actual": "phi3",
            "confidence": 0.5,
            "entity_count": 1,
            "written_count": 0,
            "queued_count": 1,
            "blocked_count": 0,
            "review_reasons": ["pending_validation_review", "pending_validation_review", "route_actual_mismatch"],
            "notes": [],
            "processing_time_ms": 20.0,
            "requested_route": "gemini",
            "retry_visibility": {"retry_detected": False, "retry_delay_seconds": None, "retry_reason": None},
        },
        {
            "document": "short_03.pdf",
            "status": "external_quota_blocked",
            "error": "429 quota exceeded; please retry in 12.5s",
            "outcome": "external_quota_blocked",
            "validation_status": "skipped_external_quota",
            "extractor": None,
            "extractor_actual": None,
            "confidence": 0.0,
            "entity_count": 0,
            "written_count": 0,
            "queued_count": 0,
            "blocked_count": 0,
            "review_reasons": [],
            "notes": [],
            "processing_time_ms": 15.0,
            "requested_route": None,
            "retry_visibility": {"retry_detected": True, "retry_delay_seconds": 12.5, "retry_reason": "external_quota"},
        },
        {
            "document": "short_04.pdf",
            "status": "error",
            "error": "boom",
            "outcome": "error",
            "validation_status": "error",
            "extractor": None,
            "extractor_actual": None,
            "confidence": 0.0,
            "entity_count": 0,
            "written_count": 0,
            "queued_count": 0,
            "blocked_count": 0,
            "review_reasons": [],
            "notes": [],
            "processing_time_ms": 5.0,
            "requested_route": None,
            "retry_visibility": {"retry_detected": False, "retry_delay_seconds": None, "retry_reason": None},
        },
    ]
    summary = build_phase12_summary(
        dataset_dir=Path("test_data/final_batch_50"),
        requested_limit=0,
        documents=documents,
        runtime_counts={"total": 3, "active": 2, "hypothesis": 0, "quarantined": 1},
        component_state={"governance_active": True},
    )

    aggregate = write_phase15_reports(tmp_path, summary)

    aggregate_path = tmp_path / "validation_aggregate.json"
    report_path = tmp_path / "validation_summary.md"

    assert aggregate_path.exists()
    assert report_path.exists()

    written_aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    assert written_aggregate["documents_attempted"] == 4
    assert written_aggregate["documents_processed"] == 2
    assert written_aggregate["documents_quota_blocked"] == 1
    assert written_aggregate["written"] == 1
    assert written_aggregate["queued_for_review"] == 1
    assert written_aggregate["hard_failures"] == 1
    assert written_aggregate["avg_confidence_processed_only"] == 0.6
    assert written_aggregate["route_distribution_actual"] == {"phi3": 1, "spacy": 1}
    assert written_aggregate["route_distribution_requested"] == {"gemini": 1, "spacy": 1, "unknown": 2}
    assert written_aggregate["top_rejection_patterns"][0] == {"pattern": "pending_validation_review", "count": 2}
    assert written_aggregate["counters"]["attempted_equals_processed_plus_quota_blocked_plus_hard_failures"] is True
    assert written_aggregate["counters"]["processed_equals_written_plus_queued_for_review_plus_other_processed_outcomes"] is True
    assert aggregate == written_aggregate

    report_text = report_path.read_text(encoding="utf-8")
    assert "Phase 15 Validation Summary" in report_text
    assert "Total documents attempted: 4" in report_text
    assert "`pending_validation_review`: 2" in report_text


def test_phase15_aggregate_does_not_regress_phase12_counts():
    documents = [
        {
            "document": "short_01.pdf",
            "status": "processed",
            "error": None,
            "outcome": "written",
            "validation_status": "accepted",
            "extractor": "spacy",
            "extractor_actual": "spacy",
            "confidence": 0.7,
            "entity_count": 2,
            "written_count": 2,
            "queued_count": 0,
            "blocked_count": 0,
            "review_reasons": [],
            "notes": [],
            "processing_time_ms": 10.0,
            "requested_route": "spacy",
            "retry_visibility": {"retry_detected": False, "retry_delay_seconds": None, "retry_reason": None},
        },
        {
            "document": "short_02.pdf",
            "status": "external_quota_blocked",
            "error": "429 quota exceeded; please retry in 8.0s",
            "outcome": "external_quota_blocked",
            "validation_status": "skipped_external_quota",
            "extractor": None,
            "extractor_actual": None,
            "confidence": 0.0,
            "entity_count": 0,
            "written_count": 0,
            "queued_count": 0,
            "blocked_count": 0,
            "review_reasons": [],
            "notes": [],
            "processing_time_ms": 8.0,
            "requested_route": None,
            "retry_visibility": {"retry_detected": True, "retry_delay_seconds": 8.0, "retry_reason": "external_quota"},
        },
    ]
    summary = build_phase12_summary(
        dataset_dir=Path("test_data/final_batch_50"),
        requested_limit=0,
        documents=documents,
        runtime_counts={"total": 2, "active": 1, "hypothesis": 0, "quarantined": 0},
        component_state={"governance_active": True},
    )
    aggregate = build_phase15_aggregate(summary)

    assert aggregate["written"] == summary["written"]
    assert aggregate["queued_for_review"] == summary["queued_for_review"]
    assert aggregate["documents_quota_blocked"] == summary["external_quota_blocked"]
    assert aggregate["hard_failures"] == summary["hard_failures"]
    assert aggregate["documents_attempted"] == summary["documents_selected"]
    assert aggregate["counters"]["attempted_equals_processed_plus_quota_blocked_plus_hard_failures"] is True
