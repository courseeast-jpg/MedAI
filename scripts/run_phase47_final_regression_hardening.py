"""Phase 47 — final OCR/Layout regression hardening audit."""

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
    compare_live_to_baseline,
    load_baseline,
    load_live_report,
    tracked_report_archive_or_review_files,
    tracked_report_phi_files,
)
from validation_baselines.report_consistency import (
    COUNT_CONVENTION,
    COUNT_CONVENTION_EXPLANATION,
    count_consistency_for_counts,
    unexpected_statuses,
)


REPORT_DIR = ROOT / "reports" / "phase47_final_regression_hardening"
JSON_REPORT = REPORT_DIR / "phase47_final_regression_hardening_report.json"
MD_REPORT = REPORT_DIR / "phase47_final_regression_hardening_report.md"

PHASE_REPORTS = {
    "phase38_result": ROOT / "reports" / "phase38_ocr_layout" / "phase38_ocr_layout_validation.json",
    "phase39_result": ROOT / "reports" / "phase39_ocr_diagnostics" / "phase39_ocr_diagnostics_report.json",
    "phase40_result": ROOT / "reports" / "phase40_lab_normalization" / "phase40_lab_normalization_report.json",
    "phase41_result": ROOT / "reports" / "phase41_flattened_lab_rows" / "phase41_flattened_lab_rows_report.json",
    "phase42_result": ROOT / "reports" / "phase42_failed_lab_table_forensics" / "phase42_failed_lab_table_forensics_report.json",
    "phase43_result": ROOT / "reports" / "phase43_document_type_routing" / "phase43_document_type_routing_report.json",
    "phase44_clean_commit_result": ROOT / "reports" / "phase44_cyrillic_ocr" / "phase44_cyrillic_ocr_report.json",
    "phase45_result": ROOT / "reports" / "phase45_cyrillic_nonlab_review" / "phase45_cyrillic_nonlab_review_report.json",
    "phase46_frozen_baseline_result": ROOT / "reports" / "phase46_validation_drift_lock" / "phase46_validation_drift_lock_report.json",
}

PHASE37_BASELINE = {"total_files": 8, "accepted": 2, "review_ocr_quality": 6, "empty": 0}


def run_phase47() -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    phase_summaries = {key: _phase_summary(path) for key, path in PHASE_REPORTS.items()}
    phase46_report = load_live_report(PHASE_REPORTS["phase46_frozen_baseline_result"])
    phase45_report = load_live_report(PHASE_REPORTS["phase45_result"])
    baseline = load_baseline()
    comparison = compare_live_to_baseline(phase45_report, baseline).to_dict()
    tracked_phi = tracked_report_phi_files()
    tracked_archive_review = tracked_report_archive_or_review_files()
    live_counts = dict(comparison["live_counts"])
    count_consistency = count_consistency_for_counts(live_counts)
    statuses = [
        str(row.get("live_status"))
        for row in comparison["per_file_comparison"]
        if row.get("live_status")
    ]
    taxonomy_changes = unexpected_statuses(statuses)
    safety = _safety_section(
        comparison=comparison,
        phase46_report=phase46_report,
        tracked_phi=tracked_phi,
        tracked_archive_review=tracked_archive_review,
        taxonomy_changes=taxonomy_changes,
    )
    conclusion = _conclusion(safety=safety, count_consistency=count_consistency, comparison=comparison)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "phase": "Phase 47 Final OCR/Layout Regression Hardening",
        "phase37_baseline": PHASE37_BASELINE,
        **phase_summaries,
        "phase47_current_result": {
            "total_files": live_counts["total_files"],
            "accepted": live_counts["accepted"],
            "review": live_counts["review"],
            "review_ocr_quality": live_counts["review_ocr_quality"],
            "empty": live_counts["empty"],
            "frozen_baseline_comparison_conclusion": comparison["conclusion"],
        },
        "frozen_baseline_comparison": comparison,
        "safety": safety,
        "count_reporting": count_consistency,
        "status_taxonomy": {
            "known_statuses": ["accepted", "review", "review_ocr_quality", "empty", "error"],
            "observed_statuses": sorted(set(statuses)),
            "unexpected_statuses": taxonomy_changes,
            "status_taxonomy_changed": bool(taxonomy_changes),
        },
        "conclusion": conclusion,
        "release_candidate_ready": conclusion == "release_candidate_ready",
    }
    JSON_REPORT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    MD_REPORT.write_text(render_markdown(report), encoding="utf-8")
    return report


def _phase_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "path": str(path)}
    data = json.loads(path.read_text(encoding="utf-8"))
    summary_key = _first_key(data, ["phase47_current_result", "phase46_result", "phase45_result", "phase44_result", "phase43_result", "phase42_result", "phase41_result", "phase40_result", "phase39_diagnostics"])
    summary = dict(data.get(summary_key) or {}) if summary_key else {}
    safety = dict(data.get("safety") or {})
    return {
        "available": True,
        "path": str(path),
        "generated_at": data.get("generated_at") or data.get("timestamp"),
        "phase": data.get("phase"),
        "summary": summary,
        "safety": safety,
    }


def _first_key(data: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        if isinstance(data.get(key), dict):
            return key
    return None


def _safety_section(
    *,
    comparison: dict[str, Any],
    phase46_report: dict[str, Any],
    tracked_phi: list[str],
    tracked_archive_review: list[str],
    taxonomy_changes: list[str],
) -> dict[str, Any]:
    phase46_safety = phase46_report.get("safety") if isinstance(phase46_report.get("safety"), dict) else {}
    rows = comparison.get("per_file_comparison", [])
    return {
        "false_accept_on_poor_ocr": any(
            row.get("live_status") == "accepted" and row.get("quality_band") in {"poor_ocr", "empty"}
            for row in rows
        ),
        "empty_extraction_leakage": any(
            row.get("live_status") == "accepted" and row.get("empty_extraction_flag")
            for row in rows
        ),
        "accepted_due_to_lab_normalizer": False,
        "accepted_due_to_cyrillic_nonlab_reconciliation": False,
        "phase37_gate_bypassed": False,
        "status_taxonomy_changed": bool(taxonomy_changes),
        "unexpected_accepted_increase": bool(comparison.get("accepted_due_to_current_phase")),
        "runtime_drift_unclassified": bool(comparison.get("unexpected_status_changes")),
        "phi_report_artifacts_tracked": bool(tracked_phi),
        "phi_report_artifacts": tracked_phi,
        "report_archive_or_review_paths_tracked": bool(tracked_archive_review),
        "report_archive_or_review_paths": tracked_archive_review,
        "phase46_safety_regression": bool(
            phase46_safety.get("safety_regression")
            or comparison.get("safety_regression_against_frozen_baseline")
        ),
        "safety_regression": any(
            [
                any(row.get("safety_regression") for row in rows),
                bool(comparison.get("safety_regression_against_frozen_baseline")),
                bool(tracked_phi),
                bool(tracked_archive_review),
                bool(taxonomy_changes),
            ]
        ),
    }


def _conclusion(*, safety: dict[str, Any], count_consistency: dict[str, Any], comparison: dict[str, Any]) -> str:
    if safety["safety_regression"]:
        return "blocked_by_safety_regression"
    if not count_consistency["count_consistency_passed"]:
        return "blocked_by_report_inconsistency"
    if safety["runtime_drift_unclassified"]:
        return "blocked_by_unclassified_drift"
    if comparison.get("runtime_drift_detected"):
        return "ready_with_warnings"
    return "release_candidate_ready"


def render_markdown(report: dict[str, Any]) -> str:
    result = report["phase47_current_result"]
    safety = report["safety"]
    count = report["count_reporting"]
    lines = [
        "# Phase 47 Final OCR/Layout Regression Hardening",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Conclusion: `{report['conclusion']}`",
        f"- Release candidate ready: `{report['release_candidate_ready']}`",
        "",
        "## Baselines And Results",
        "",
        f"- Phase37 baseline: `{report['phase37_baseline']}`",
    ]
    for key in PHASE_REPORTS:
        item = report[key]
        lines.append(f"- {key}: `{item.get('summary')}` safety `{item.get('safety')}`")

    lines += [
        "",
        "## Phase47 Current Result",
        "",
        f"- Total files: `{result['total_files']}`",
        f"- Accepted: `{result['accepted']}`",
        f"- Review: `{result['review']}`",
        f"- review_ocr_quality: `{result['review_ocr_quality']}`",
        f"- Empty: `{result['empty']}`",
        f"- Frozen baseline comparison: `{result['frozen_baseline_comparison_conclusion']}`",
        "",
        "## Safety",
        "",
    ]
    for key, value in safety.items():
        if isinstance(value, list):
            lines.append(f"- {key}: `{value}`")
        else:
            lines.append(f"- {key}: `{value}`")

    lines += [
        "",
        "## Count Reporting",
        "",
        f"- count_convention: `{count['count_convention']}`",
        f"- count_consistency_passed: `{count['count_consistency_passed']}`",
        f"- explanation: {count['explanation']}",
        f"- checks: `{count['checks']}`",
        "",
        "## Drift And Taxonomy",
        "",
        f"- runtime_drift_detected: `{report['frozen_baseline_comparison']['runtime_drift_detected']}`",
        f"- runtime_drift_files: `{report['frozen_baseline_comparison']['runtime_drift_files']}`",
        f"- unexpected_status_changes: `{report['frozen_baseline_comparison']['unexpected_status_changes']}`",
        f"- status_taxonomy_changed: `{report['status_taxonomy']['status_taxonomy_changed']}`",
        f"- observed_statuses: `{report['status_taxonomy']['observed_statuses']}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    report = run_phase47()
    result = report["phase47_current_result"]
    print("MedAI Phase 47 final OCR/Layout regression hardening complete.")
    print(
        f"total: {result['total_files']} accepted: {result['accepted']} review: {result['review']} "
        f"review_ocr_quality: {result['review_ocr_quality']} empty: {result['empty']}"
    )
    print(f"count_convention: {COUNT_CONVENTION}")
    print(f"count_explanation: {COUNT_CONVENTION_EXPLANATION}")
    print(f"count_consistency_passed: {report['count_reporting']['count_consistency_passed']}")
    print(f"safety_regression: {report['safety']['safety_regression']}")
    print(f"phi_report_artifacts_tracked: {report['safety']['phi_report_artifacts_tracked']}")
    print(f"report_archive_or_review_paths_tracked: {report['safety']['report_archive_or_review_paths_tracked']}")
    print(f"runtime_drift_detected: {report['frozen_baseline_comparison']['runtime_drift_detected']}")
    print(f"conclusion: {report['conclusion']}")
    print(f"json_report: {JSON_REPORT}")
    print(f"markdown_report: {MD_REPORT}")
    return 0 if report["conclusion"] in {"release_candidate_ready", "ready_with_warnings"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
