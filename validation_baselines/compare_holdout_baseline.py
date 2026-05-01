"""Compare live holdout validation output against a frozen benchmark."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = ROOT / "validation_baselines" / "holdout_phase45_baseline.json"
KNOWN_STATUSES = {"accepted", "review", "review_ocr_quality", "empty", "error"}


@dataclass(frozen=True)
class BaselineComparison:
    baseline: dict[str, Any]
    live_counts: dict[str, int]
    count_comparison: dict[str, dict[str, int]]
    per_file: list[dict[str, Any]]
    runtime_drift_detected: bool
    runtime_drift_files: list[str]
    safety_regression_against_frozen_baseline: bool
    unexpected_status_changes: list[str]
    accepted_due_to_current_phase: list[str]
    tracked_phi_report_files: list[str]
    report_archive_or_review_paths_tracked: bool
    conclusion: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "frozen_baseline": self.baseline,
            "live_counts": self.live_counts,
            "count_comparison": self.count_comparison,
            "per_file_comparison": self.per_file,
            "runtime_drift_detected": self.runtime_drift_detected,
            "runtime_drift_files": self.runtime_drift_files,
            "safety_regression_against_frozen_baseline": self.safety_regression_against_frozen_baseline,
            "unexpected_status_changes": self.unexpected_status_changes,
            "accepted_due_to_current_phase": self.accepted_due_to_current_phase,
            "tracked_phi_report_files": self.tracked_phi_report_files,
            "report_archive_or_review_paths_tracked": self.report_archive_or_review_paths_tracked,
            "conclusion": self.conclusion,
        }


def load_baseline(path: Path = BASELINE_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_live_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_live_to_baseline(
    live_report: dict[str, Any],
    baseline: dict[str, Any] | None = None,
    *,
    tracked_files: list[str] | None = None,
) -> BaselineComparison:
    baseline = baseline or load_baseline()
    live_results = _live_results(live_report)
    rows = [_compare_file(filename, expected, live_results.get(filename)) for filename, expected in baseline["files"].items()]

    for filename in sorted(set(live_results) - set(baseline["files"])):
        rows.append(
            {
                "filename": filename,
                "expected_status": None,
                "live_status": _status_for(live_results[filename]),
                "classification": "taxonomy_change",
                "drift_sensitive": False,
                "safety_regression": False,
                "reason": "file_not_present_in_frozen_baseline",
            }
        )

    live_counts = _count_live(live_results)
    count_comparison = _count_comparison(baseline, live_counts)
    tracked_phi_files = tracked_files if tracked_files is not None else tracked_report_phi_files()
    phi_tracked = bool(tracked_phi_files)

    accepted_due_to_current_phase = [
        row["filename"]
        for row in rows
        if row.get("live_status") == "accepted"
        and row.get("expected_status") != "accepted"
        and row.get("classification") != "acceptable_runtime_drift"
    ]
    unapproved_accepted_increase = live_counts["accepted"] > int(baseline["expected_counts"]["accepted"]) and bool(accepted_due_to_current_phase)
    safety_regression = phi_tracked or unapproved_accepted_increase or any(row.get("safety_regression") for row in rows)
    unexpected_changes = [
        row["filename"]
        for row in rows
        if row["classification"] in {"unexpected_status_change", "taxonomy_change"}
    ]
    runtime_drift_files = [
        row["filename"] for row in rows if row["classification"] == "acceptable_runtime_drift"
    ]

    conclusion = _conclusion(
        safety_regression=safety_regression,
        unexpected_changes=unexpected_changes,
        runtime_drift_files=runtime_drift_files,
    )
    return BaselineComparison(
        baseline=baseline,
        live_counts=live_counts,
        count_comparison=count_comparison,
        per_file=rows,
        runtime_drift_detected=bool(runtime_drift_files),
        runtime_drift_files=runtime_drift_files,
        safety_regression_against_frozen_baseline=safety_regression,
        unexpected_status_changes=unexpected_changes,
        accepted_due_to_current_phase=accepted_due_to_current_phase,
        tracked_phi_report_files=tracked_phi_files,
        report_archive_or_review_paths_tracked=phi_tracked,
        conclusion=conclusion,
    )


def tracked_report_phi_files(root: Path = ROOT) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "reports/**/archive/*", "reports/**/review/*"],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    return [
        line.strip()
        for line in (result.stdout or "").splitlines()
        if line.strip() and not line.strip().endswith("/.gitkeep")
    ]


def _compare_file(filename: str, expected: dict[str, Any], live: dict[str, Any] | None) -> dict[str, Any]:
    if live is None:
        return {
            "filename": filename,
            "expected_status": expected.get("expected_status"),
            "live_status": None,
            "classification": "unexpected_status_change",
            "drift_sensitive": bool(expected.get("drift_sensitive")),
            "safety_regression": False,
            "reason": "missing_from_live_validation",
            "expected_reason_codes": list(expected.get("reason_codes") or []),
            "live_reason_codes": [],
        }

    expected_status = str(expected.get("expected_status"))
    live_status = _status_for(live)
    reason_codes = list(live.get("reason_codes") or live.get("classification_reason_codes") or [])
    empty_flag = bool(live.get("empty_extraction_flag", False))
    quality_band = str(live.get("ocr_layout_quality_band") or live.get("input_quality_band") or "")
    classification, reason = _classify_status_change(
        expected=expected,
        expected_status=expected_status,
        live_status=live_status,
        empty_flag=empty_flag,
        quality_band=quality_band,
    )
    safety = _is_safety_regression(
        expected_status=expected_status,
        live_status=live_status,
        empty_flag=empty_flag,
        quality_band=quality_band,
        classification=classification,
    )
    return {
        "filename": filename,
        "expected_status": expected_status,
        "live_status": live_status,
        "classification": classification,
        "drift_sensitive": bool(expected.get("drift_sensitive")),
        "drift_reason": expected.get("drift_reason"),
        "safety_regression": safety,
        "reason": reason,
        "quality_band": quality_band,
        "empty_extraction_flag": empty_flag,
        "expected_reason_codes": list(expected.get("reason_codes") or []),
        "live_reason_codes": reason_codes,
    }


def _classify_status_change(
    *,
    expected: dict[str, Any],
    expected_status: str,
    live_status: str,
    empty_flag: bool,
    quality_band: str,
) -> tuple[str, str]:
    if live_status not in KNOWN_STATUSES:
        return "taxonomy_change", "live_status_not_in_known_taxonomy"
    if live_status == expected_status:
        return "expected_match", "status_matches_frozen_baseline"

    drift_statuses = set(expected.get("acceptable_drift_statuses") or [])
    if (
        expected.get("drift_sensitive")
        and {expected_status, live_status}.issubset({"review", "accepted"})
        and live_status in drift_statuses
        and not empty_flag
        and quality_band not in {"poor_ocr", "empty"}
    ):
        return "acceptable_runtime_drift", str(expected.get("drift_reason") or "drift_sensitive_file")

    if expected_status == "review_ocr_quality" and live_status == "accepted":
        return "safety_regression", "review_ocr_quality_became_accepted_without_approved_phase_reason"
    if live_status == "accepted":
        return "unexpected_status_change", "unapproved_accepted_increase"
    if expected_status == "review_ocr_quality" and live_status == "review":
        return "improvement_candidate", "review_ocr_quality_moved_to_review_without_acceptance"
    return "unexpected_status_change", "status_changed_without_drift_allowance"


def _is_safety_regression(
    *,
    expected_status: str,
    live_status: str,
    empty_flag: bool,
    quality_band: str,
    classification: str,
) -> bool:
    if live_status == "accepted" and quality_band in {"poor_ocr", "empty"}:
        return True
    if live_status == "accepted" and empty_flag:
        return True
    if expected_status == "review_ocr_quality" and live_status == "accepted":
        return True
    return classification == "safety_regression"


def _live_results(live_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    items = live_report.get("results") or []
    return {
        str(item.get("filename")): item
        for item in items
        if isinstance(item, dict) and item.get("filename")
    }


def _status_for(item: dict[str, Any]) -> str:
    for key in ("final_status_after_phase45", "final_status", "final_status_after_lab_normalization", "status"):
        value = item.get(key)
        if value:
            return str(value)
    return "unknown"


def _count_live(live_results: dict[str, dict[str, Any]]) -> dict[str, int]:
    statuses = [_status_for(item) for item in live_results.values()]
    return {
        "total_files": len(live_results),
        "accepted": statuses.count("accepted"),
        "review": sum(1 for status in statuses if status in {"review", "review_ocr_quality"}),
        "review_ocr_quality": statuses.count("review_ocr_quality"),
        "empty": sum(1 for item in live_results.values() if bool(item.get("empty_extraction_flag", False))),
    }


def _count_comparison(baseline: dict[str, Any], live_counts: dict[str, int]) -> dict[str, dict[str, int]]:
    expected = {"total_files": int(baseline.get("total_files") or 0), **baseline["expected_counts"]}
    return {
        key: {
            "expected": int(expected.get(key) or 0),
            "live": int(live_counts.get(key) or 0),
            "delta": int(live_counts.get(key) or 0) - int(expected.get(key) or 0),
        }
        for key in ("total_files", "accepted", "review", "review_ocr_quality", "empty")
    }


def _conclusion(
    *,
    safety_regression: bool,
    unexpected_changes: list[str],
    runtime_drift_files: list[str],
) -> str:
    if safety_regression:
        return "blocked_by_safety_regression"
    if unexpected_changes:
        return "blocked_by_unexpected_drift"
    if runtime_drift_files:
        return "drift_detected_but_safe"
    return "deterministic_lock_ready"
