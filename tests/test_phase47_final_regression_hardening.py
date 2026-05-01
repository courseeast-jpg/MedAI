from __future__ import annotations

import json
from pathlib import Path

from scripts.run_phase47_final_regression_hardening import run_phase47
from validation_baselines.compare_holdout_baseline import (
    compare_live_to_baseline,
    load_baseline,
    tracked_report_archive_or_review_files,
    tracked_report_phi_files,
)
from validation_baselines.report_consistency import count_consistency_for_counts, unexpected_statuses


def _live_from_baseline(baseline: dict) -> dict:
    return {
        "results": [
            {
                "filename": filename,
                "final_status_after_phase45": row["expected_status"],
                "empty_extraction_flag": row.get("expected_empty_extraction_flag", False),
                "ocr_layout_quality_band": "usable_with_review",
                "reason_codes": list(row.get("reason_codes") or []),
            }
            for filename, row in baseline["files"].items()
        ]
    }


def test_phase47_poor_ocr_cannot_be_accepted():
    baseline = load_baseline()
    live = _live_from_baseline(baseline)
    live["results"][0]["final_status_after_phase45"] = "accepted"
    live["results"][0]["ocr_layout_quality_band"] = "poor_ocr"

    comparison = compare_live_to_baseline(live, baseline, tracked_files=[])

    assert comparison.safety_regression_against_frozen_baseline is True
    assert comparison.conclusion == "blocked_by_safety_regression"


def test_phase47_empty_extraction_cannot_be_accepted():
    baseline = load_baseline()
    live = _live_from_baseline(baseline)
    live["results"][0]["final_status_after_phase45"] = "accepted"
    live["results"][0]["empty_extraction_flag"] = True

    comparison = compare_live_to_baseline(live, baseline, tracked_files=[])

    assert comparison.safety_regression_against_frozen_baseline is True
    assert comparison.conclusion == "blocked_by_safety_regression"


def test_phase47_lab_normalizer_and_cyrillic_reconciliation_do_not_accept():
    phase46 = json.loads(Path("reports/phase46_validation_drift_lock/phase46_validation_drift_lock_report.json").read_text(encoding="utf-8"))
    rows = phase46["per_file_comparison"]

    assert not any(
        row["live_status"] == "accepted" and "lab_table_recovered" in row.get("live_reason_codes", [])
        for row in rows
    )
    assert not any(
        row["live_status"] == "accepted" and "cyrillic_non_lab_document_review" in row.get("live_reason_codes", [])
        for row in rows
    )


def test_phase47_drift_sensitive_files_are_not_phase_improvements():
    baseline = load_baseline()
    live = _live_from_baseline(baseline)
    for item in live["results"]:
        if item["filename"] == "Results 1.pdf":
            item["final_status_after_phase45"] = "accepted"

    comparison = compare_live_to_baseline(live, baseline, tracked_files=[])
    row = next(row for row in comparison.per_file if row["filename"] == "Results 1.pdf")

    assert row["classification"] == "acceptable_runtime_drift"
    assert "Results 1.pdf" not in comparison.accepted_due_to_current_phase
    assert comparison.conclusion == "drift_detected_but_safe"


def test_phase47_phi_report_artifacts_are_not_tracked():
    assert tracked_report_phi_files() == []
    assert tracked_report_archive_or_review_files() == []


def test_phase47_frozen_baseline_comparison_still_works():
    baseline = load_baseline()
    comparison = compare_live_to_baseline(_live_from_baseline(baseline), baseline, tracked_files=[])

    assert comparison.conclusion == "deterministic_lock_ready"
    assert comparison.live_counts["accepted"] == 2
    assert comparison.live_counts["review_ocr_quality"] == 2


def test_phase47_status_taxonomy_has_not_changed():
    assert unexpected_statuses(["accepted", "review", "review_ocr_quality", "empty", "error"]) == []
    assert unexpected_statuses(["accepted", "auto_accepted_v2"]) == ["auto_accepted_v2"]


def test_phase47_count_convention_is_overlapping_and_consistent():
    result = count_consistency_for_counts(
        {"total_files": 8, "accepted": 2, "review": 6, "review_ocr_quality": 2, "empty": 2}
    )

    assert result["count_convention"] == "overlapping_review_total_with_review_ocr_quality_and_empty_subsets"
    assert result["count_consistency_passed"] is True
    assert result["checks"]["accepted_plus_review_equals_total"] is True


def test_phase47_count_convention_detects_inconsistency():
    result = count_consistency_for_counts(
        {"total_files": 8, "accepted": 3, "review": 6, "review_ocr_quality": 2, "empty": 2}
    )

    assert result["count_consistency_passed"] is False


def test_phase47_benchmark_lock_report_remains_deterministic_ready():
    report = json.loads(Path("reports/phase46_validation_drift_lock/phase46_validation_drift_lock_report.json").read_text(encoding="utf-8"))

    assert report["conclusion"] == "deterministic_lock_ready"
    assert report["runtime_drift_detected"] is False
    assert report["safety_regression_against_frozen_baseline"] is False


def test_phase47_final_audit_report_can_be_built_without_pipeline_changes():
    report = run_phase47()

    assert report["conclusion"] == "release_candidate_ready"
    assert report["release_candidate_ready"] is True
    assert report["safety"]["safety_regression"] is False
    assert report["count_reporting"]["count_consistency_passed"] is True
