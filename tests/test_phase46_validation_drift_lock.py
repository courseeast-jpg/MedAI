from __future__ import annotations

from pathlib import Path

from validation_baselines.compare_holdout_baseline import (
    compare_live_to_baseline,
    load_baseline,
    tracked_report_phi_files,
)


def _live_from_baseline(baseline: dict) -> dict:
    return {
        "results": [
            {
                "filename": filename,
                "final_status_after_phase45": item["expected_status"],
                "empty_extraction_flag": item.get("expected_empty_extraction_flag", False),
                "ocr_layout_quality_band": "usable_with_review",
                "reason_codes": list(item.get("reason_codes") or []),
            }
            for filename, item in baseline["files"].items()
        ]
    }


def test_phase46_baseline_manifest_loads_correctly():
    baseline = load_baseline()

    assert baseline["baseline_name"] == "holdout_phase45_baseline"
    assert baseline["commit"] == "e6e437d"
    assert baseline["total_files"] == 8
    assert baseline["expected_counts"]["accepted"] == 2
    assert baseline["files"]["Results 1.pdf"]["drift_sensitive"] is True
    assert baseline["files"]["Results 1.pdf"]["drift_reason"] == "near_threshold_upstream_extractor_variability"


def test_phase46_exact_match_comparison_is_deterministic_lock_ready():
    baseline = load_baseline()
    comparison = compare_live_to_baseline(_live_from_baseline(baseline), baseline, tracked_files=[])

    assert comparison.conclusion == "deterministic_lock_ready"
    assert comparison.runtime_drift_detected is False
    assert comparison.safety_regression_against_frozen_baseline is False
    assert {row["classification"] for row in comparison.per_file} == {"expected_match"}


def test_phase46_results_1_accepted_drift_is_acceptable_runtime_drift():
    baseline = load_baseline()
    live = _live_from_baseline(baseline)
    for item in live["results"]:
        if item["filename"] == "Results 1.pdf":
            item["final_status_after_phase45"] = "accepted"

    comparison = compare_live_to_baseline(live, baseline, tracked_files=[])
    row = next(row for row in comparison.per_file if row["filename"] == "Results 1.pdf")

    assert row["classification"] == "acceptable_runtime_drift"
    assert comparison.runtime_drift_files == ["Results 1.pdf"]
    assert comparison.conclusion == "drift_detected_but_safe"
    assert comparison.safety_regression_against_frozen_baseline is False


def test_phase46_poor_ocr_accepted_is_safety_regression():
    baseline = load_baseline()
    live = _live_from_baseline(baseline)
    for item in live["results"]:
        if item["filename"] == "Test Results 5.pdf":
            item["final_status_after_phase45"] = "accepted"
            item["ocr_layout_quality_band"] = "poor_ocr"

    comparison = compare_live_to_baseline(live, baseline, tracked_files=[])
    row = next(row for row in comparison.per_file if row["filename"] == "Test Results 5.pdf")

    assert row["classification"] == "safety_regression"
    assert row["safety_regression"] is True
    assert comparison.conclusion == "blocked_by_safety_regression"


def test_phase46_unapproved_accepted_increase_is_blocked():
    baseline = load_baseline()
    live = _live_from_baseline(baseline)
    for item in live["results"]:
        if item["filename"] == "Results 2.pdf":
            item["final_status_after_phase45"] = "accepted"

    comparison = compare_live_to_baseline(live, baseline, tracked_files=[])
    row = next(row for row in comparison.per_file if row["filename"] == "Results 2.pdf")

    assert row["classification"] == "unexpected_status_change"
    assert comparison.safety_regression_against_frozen_baseline is True
    assert comparison.conclusion == "blocked_by_safety_regression"


def test_phase46_tracked_phi_report_files_fail_safety_check():
    baseline = load_baseline()
    comparison = compare_live_to_baseline(
        _live_from_baseline(baseline),
        baseline,
        tracked_files=["reports/phase46_validation_drift_lock/archive/patient.pdf"],
    )

    assert comparison.report_archive_or_review_paths_tracked is True
    assert comparison.safety_regression_against_frozen_baseline is True
    assert comparison.conclusion == "blocked_by_safety_regression"


def test_phase46_report_archive_and_review_paths_remain_untracked_or_gitkeep_only():
    tracked = tracked_report_phi_files()

    assert not any(path.lower().endswith(".pdf") for path in tracked)
    assert set(tracked).issubset(
        {
            "reports/batch_validation/archive/.gitkeep",
            "reports/batch_validation/review/.gitkeep",
        }
    )


def test_phase46_gitignore_keeps_report_archive_and_review_outputs_ignored():
    gitignore = Path(".gitignore").read_text(encoding="utf-8")

    assert "reports/**/archive/" in gitignore
    assert "reports/**/review/" in gitignore


def test_phase46_phase45_safety_baseline_stays_stable():
    baseline = load_baseline()
    comparison = compare_live_to_baseline(_live_from_baseline(baseline), baseline, tracked_files=[])

    assert comparison.live_counts["accepted"] == baseline["expected_counts"]["accepted"]
    assert comparison.live_counts["review_ocr_quality"] == baseline["expected_counts"]["review_ocr_quality"]
    assert comparison.safety_regression_against_frozen_baseline is False
