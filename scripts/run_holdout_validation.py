from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_batch_validation as batch


HOLDOUT_INPUT_DIR = ROOT / "holdout_validation_input"
HOLDOUT_REPORT_DIR = ROOT / "reports" / "holdout_validation"
HOLDOUT_ARCHIVE_DIR = HOLDOUT_REPORT_DIR / "archive"
HOLDOUT_REVIEW_DIR = HOLDOUT_REPORT_DIR / "review"
HOLDOUT_ERROR_DIR = HOLDOUT_REPORT_DIR / "error"
HOLDOUT_JSON_REPORT = HOLDOUT_REPORT_DIR / "latest_holdout_validation.json"
HOLDOUT_MD_REPORT = HOLDOUT_REPORT_DIR / "latest_holdout_validation.md"
HOLDOUT_REVIEW_AUDIT_JSON = HOLDOUT_REPORT_DIR / "review_audit.json"
HOLDOUT_REVIEW_AUDIT_MD = HOLDOUT_REPORT_DIR / "review_audit.md"
HOLDOUT_COMPARISON_REPORT = HOLDOUT_REPORT_DIR / "comparison_to_baseline.md"
BASELINE_JSON_REPORT = ROOT / "reports" / "baseline_phase33" / "baseline_metrics.json"


def configure_holdout_paths() -> None:
    batch.REAL_VALIDATION_INPUT_DIR = HOLDOUT_INPUT_DIR
    batch.BATCH_REPORT_DIR = HOLDOUT_REPORT_DIR
    batch.BATCH_ARCHIVE_DIR = HOLDOUT_ARCHIVE_DIR
    batch.BATCH_REVIEW_DIR = HOLDOUT_REVIEW_DIR
    batch.BATCH_ERROR_DIR = HOLDOUT_ERROR_DIR
    batch.BATCH_JSON_REPORT = HOLDOUT_JSON_REPORT
    batch.BATCH_MD_REPORT = HOLDOUT_MD_REPORT
    batch.REVIEW_AUDIT_JSON_REPORT = HOLDOUT_REVIEW_AUDIT_JSON
    batch.REVIEW_AUDIT_MD_REPORT = HOLDOUT_REVIEW_AUDIT_MD


def ensure_holdout_dirs() -> None:
    for path in (
        HOLDOUT_INPUT_DIR,
        HOLDOUT_REPORT_DIR,
        HOLDOUT_ARCHIVE_DIR,
        HOLDOUT_REVIEW_DIR,
        HOLDOUT_ERROR_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def run_holdout_validation(pipeline=None) -> dict[str, Any]:
    ensure_holdout_dirs()
    configure_holdout_paths()
    summary = batch.run_batch_validation(pipeline=pipeline)
    write_comparison_to_baseline(summary)
    return summary


def load_baseline() -> dict[str, Any]:
    return json.loads(BASELINE_JSON_REPORT.read_text(encoding="utf-8"))


def write_comparison_to_baseline(summary: dict[str, Any]) -> Path:
    baseline = load_baseline()
    baseline_total = int(baseline.get("total_files") or 0)
    holdout_total = int(summary.get("total_files") or 0)
    rows = [
        ("accepted", int(baseline.get("accepted") or 0), int(summary.get("accepted_count") or 0)),
        ("review", int(baseline.get("review") or 0), int(summary.get("review_count") or 0)),
        ("empty", int(baseline.get("empty") or 0), int(summary.get("empty_extraction_count") or 0)),
    ]

    lines = [
        "# Holdout Validation vs Phase 33 Baseline",
        "",
        f"- Baseline commit: `{baseline.get('commit_hash')}`",
        f"- Baseline total files: `{baseline_total}`",
        f"- Holdout total files: `{holdout_total}`",
        "",
        "| Metric | Baseline % | Holdout % | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, baseline_count, holdout_count in rows:
        baseline_pct = percent(baseline_count, baseline_total)
        holdout_pct = percent(holdout_count, holdout_total)
        lines.append(f"| {label} | {baseline_pct:.2f}% | {holdout_pct:.2f}% | {holdout_pct - baseline_pct:+.2f}% |")

    HOLDOUT_COMPARISON_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return HOLDOUT_COMPARISON_REPORT


def percent(value: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return (value / total) * 100


def main() -> int:
    summary = run_holdout_validation()
    if summary["total_files"] == 0:
        print("No supported PDF or TXT files found in holdout_validation_input/. Nothing to validate.")
    else:
        print("MedAI holdout validation complete.")
    print(f"total: {summary['total_files']}")
    print(f"accepted: {summary['accepted_count']}")
    print(f"review: {summary['review_count']}")
    print(f"errors: {summary['error_count']}")
    print(f"empty_extractions: {summary['empty_extraction_count']}")
    print(f"json_report: {HOLDOUT_JSON_REPORT}")
    print(f"markdown_report: {HOLDOUT_MD_REPORT}")
    print(f"comparison_report: {HOLDOUT_COMPARISON_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
