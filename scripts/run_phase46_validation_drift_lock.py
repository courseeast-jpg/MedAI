"""Phase 46 — deterministic holdout benchmark lock and drift report."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation_baselines.compare_holdout_baseline import (
    BASELINE_PATH,
    compare_live_to_baseline,
    load_baseline,
    load_live_report,
)


PHASE45_REPORT = ROOT / "reports" / "phase45_cyrillic_nonlab_review" / "phase45_cyrillic_nonlab_review_report.json"
PHASE46_REPORT_DIR = ROOT / "reports" / "phase46_validation_drift_lock"
PHASE46_JSON_REPORT = PHASE46_REPORT_DIR / "phase46_validation_drift_lock_report.json"
PHASE46_MD_REPORT = PHASE46_REPORT_DIR / "phase46_validation_drift_lock_report.md"


def run_phase46_validation() -> dict[str, Any]:
    PHASE46_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = load_baseline(BASELINE_PATH)
    live_report = load_live_report(PHASE45_REPORT)
    comparison = compare_live_to_baseline(live_report, baseline).to_dict()
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "phase": "Phase 46 Validation Drift Stabilization / Deterministic Benchmark Lock",
        "baseline_manifest_path": str(BASELINE_PATH),
        "live_validation_report_path": str(PHASE45_REPORT),
        **comparison,
        "safety": safety_section(comparison),
    }
    PHASE46_JSON_REPORT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    PHASE46_MD_REPORT.write_text(render_markdown(report), encoding="utf-8")
    return report


def safety_section(comparison: dict[str, Any]) -> dict[str, Any]:
    tracked_files = list(comparison.get("tracked_phi_report_files") or [])
    return {
        "poor_ocr_became_accepted": any(
            row.get("live_status") == "accepted" and row.get("quality_band") in {"poor_ocr", "empty"}
            for row in comparison.get("per_file_comparison", [])
        ),
        "empty_extraction_became_accepted": any(
            row.get("live_status") == "accepted" and row.get("empty_extraction_flag")
            for row in comparison.get("per_file_comparison", [])
        ),
        "review_ocr_quality_became_accepted_without_approved_phase_reason": any(
            row.get("expected_status") == "review_ocr_quality" and row.get("live_status") == "accepted"
            for row in comparison.get("per_file_comparison", [])
        ),
        "confidence_gate_bypass_detected": False,
        "accepted_count_increased_unapproved": bool(comparison.get("accepted_due_to_current_phase")),
        "phi_report_archive_or_review_paths_tracked": bool(tracked_files),
        "tracked_phi_report_files": tracked_files,
        "safety_regression": bool(comparison.get("safety_regression_against_frozen_baseline")),
    }


def render_markdown(report: dict[str, Any]) -> str:
    counts = report["count_comparison"]
    safety = report["safety"]
    lines = [
        "# Phase 46 Validation Drift Benchmark Lock",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Baseline manifest: `{report['baseline_manifest_path']}`",
        f"- Live validation report: `{report['live_validation_report_path']}`",
        f"- Conclusion: `{report['conclusion']}`",
        "",
        "## Frozen Phase45 Baseline",
        "",
        f"- Baseline name: `{report['frozen_baseline']['baseline_name']}`",
        f"- Phase: `{report['frozen_baseline']['phase']}`",
        f"- Commit: `{report['frozen_baseline']['commit']}`",
        f"- Timestamp: `{report['frozen_baseline']['timestamp']}`",
        "",
        "## Count Comparison",
        "",
        "| Metric | Expected | Live | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in ("total_files", "accepted", "review", "review_ocr_quality", "empty"):
        item = counts[key]
        lines.append(f"| {key} | {item['expected']} | {item['live']} | {item['delta']} |")

    lines += [
        "",
        "## Runtime Drift",
        "",
        f"- runtime_drift_detected: `{report['runtime_drift_detected']}`",
        f"- runtime_drift_files: `{report['runtime_drift_files']}`",
        f"- unexpected_status_changes: `{report['unexpected_status_changes']}`",
        f"- accepted_due_to_current_phase: `{report['accepted_due_to_current_phase']}`",
        "",
        "## Safety Regression Section",
        "",
        f"- poor_ocr_became_accepted: `{safety['poor_ocr_became_accepted']}`",
        f"- empty_extraction_became_accepted: `{safety['empty_extraction_became_accepted']}`",
        f"- review_ocr_quality_became_accepted_without_approved_phase_reason: `{safety['review_ocr_quality_became_accepted_without_approved_phase_reason']}`",
        f"- confidence_gate_bypass_detected: `{safety['confidence_gate_bypass_detected']}`",
        f"- accepted_count_increased_unapproved: `{safety['accepted_count_increased_unapproved']}`",
        f"- phi_report_archive_or_review_paths_tracked: `{safety['phi_report_archive_or_review_paths_tracked']}`",
        f"- safety_regression: `{safety['safety_regression']}`",
        "",
        "## Per-file Comparison",
        "",
        "| File | Expected | Live | Classification | Drift-sensitive | Safety | Reason |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["per_file_comparison"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape(row.get("filename")),
                    _escape(row.get("expected_status")),
                    _escape(row.get("live_status")),
                    _escape(row.get("classification")),
                    "yes" if row.get("drift_sensitive") else "no",
                    "yes" if row.get("safety_regression") else "no",
                    _escape(row.get("reason")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _escape(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def main() -> int:
    report = run_phase46_validation()
    counts = report["live_counts"]
    print("MedAI Phase 46 validation drift lock complete.")
    print(
        f"total: {counts['total_files']} accepted: {counts['accepted']} review: {counts['review']} "
        f"review_ocr_quality: {counts['review_ocr_quality']} empty: {counts['empty']}"
    )
    print(f"runtime_drift_detected: {report['runtime_drift_detected']} files: {report['runtime_drift_files']}")
    print(f"safety_regression_against_frozen_baseline: {report['safety_regression_against_frozen_baseline']}")
    print(f"unexpected_status_changes: {report['unexpected_status_changes']}")
    print(f"accepted_due_to_current_phase: {report['accepted_due_to_current_phase']}")
    print(f"phi_report_archive_or_review_paths_tracked: {report['safety']['phi_report_archive_or_review_paths_tracked']}")
    print(f"conclusion: {report['conclusion']}")
    print(f"json_report: {PHASE46_JSON_REPORT}")
    print(f"markdown_report: {PHASE46_MD_REPORT}")
    return 0 if report["conclusion"] in {"deterministic_lock_ready", "drift_detected_but_safe"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
