from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_batch_validation as batch


HOLDOUT_INPUT_DIR = ROOT / "holdout_validation_input"
PHASE38_REPORT_DIR = ROOT / "reports" / "phase38_ocr_layout"
PHASE38_ARCHIVE_DIR = PHASE38_REPORT_DIR / "archive"
PHASE38_REVIEW_DIR = PHASE38_REPORT_DIR / "review"
PHASE38_ERROR_DIR = PHASE38_REPORT_DIR / "error"
PHASE38_JSON_REPORT = PHASE38_REPORT_DIR / "phase38_ocr_layout_validation.json"
PHASE38_MD_REPORT = PHASE38_REPORT_DIR / "phase38_ocr_layout_validation.md"

PHASE37_HOLDOUT_BASELINE = {
    "total_files": 8,
    "accepted": 2,
    "review_ocr_quality": 6,
    "empty": 0,
}


def configure_phase38_paths() -> None:
    batch.REAL_VALIDATION_INPUT_DIR = HOLDOUT_INPUT_DIR
    batch.BATCH_REPORT_DIR = PHASE38_REPORT_DIR
    batch.BATCH_ARCHIVE_DIR = PHASE38_ARCHIVE_DIR
    batch.BATCH_REVIEW_DIR = PHASE38_REVIEW_DIR
    batch.BATCH_ERROR_DIR = PHASE38_ERROR_DIR
    batch.BATCH_JSON_REPORT = PHASE38_REPORT_DIR / "latest_phase38_batch_validation.json"
    batch.BATCH_MD_REPORT = PHASE38_REPORT_DIR / "latest_phase38_batch_validation.md"
    batch.REVIEW_AUDIT_JSON_REPORT = PHASE38_REPORT_DIR / "review_audit.json"
    batch.REVIEW_AUDIT_MD_REPORT = PHASE38_REPORT_DIR / "review_audit.md"


def run_phase38_validation() -> dict[str, Any]:
    configure_phase38_paths()
    PHASE38_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = batch.run_batch_validation()
    report = build_phase38_report(summary)
    PHASE38_JSON_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    PHASE38_MD_REPORT.write_text(render_markdown(report), encoding="utf-8")
    return report


def build_phase38_report(summary: dict[str, Any]) -> dict[str, Any]:
    results = list(summary.get("results") or [])
    review_ocr_quality = sum(1 for item in results if item.get("status") == "review_ocr_quality")
    review_total = sum(1 for item in results if item.get("status") in {"review", "review_ocr_quality"})
    empty = int(summary.get("empty_extraction_count") or 0)
    accepted = int(summary.get("accepted_count") or 0)
    previous_ocr_review_files = {
        str(item.get("filename"))
        for item in _load_phase37_holdout_results()
        if item.get("status") == "review_ocr_quality"
    }
    per_file = [phase38_file_row(item, previous_ocr_review_files) for item in results]
    improved = sum(1 for item in per_file if item["phase37_review_ocr_quality_improved"])
    safety_regression = any(
        item["status"] == "accepted" and item["input_quality_band"] in {"poor_ocr", "empty"}
        for item in per_file
    )
    mismatch_count = sum(1 for item in per_file if item["ocr_status_mismatch"])

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "phase": "Phase 38 v2 OCR/Layout Engine Foundation",
        "source_input_dir": str(HOLDOUT_INPUT_DIR),
        "phase37_holdout_baseline": dict(PHASE37_HOLDOUT_BASELINE),
        "total_files": int(summary.get("total_files") or 0),
        "accepted": accepted,
        "review": review_total,
        "review_ocr_quality": review_ocr_quality,
        "empty": empty,
        "ocr_layout_route_summary": dict(summary.get("ocr_layout_route_summary") or {}),
        "ocr_layout_quality_summary": dict(summary.get("ocr_layout_quality_summary") or {}),
        "phase37_review_ocr_quality_improved_count": improved,
        "ocr_status_mismatch_count": mismatch_count,
        "input_quality_improved": improved > 0,
        "safety_regression": safety_regression,
        "safety_regression_reason": None
        if not safety_regression
        else "A poor or empty OCR/Layout quality band reached accepted status.",
        "results": per_file,
        "raw_batch_report": str(batch.BATCH_JSON_REPORT),
        "raw_batch_markdown": str(batch.BATCH_MD_REPORT),
    }


def phase38_file_row(item: dict[str, Any], previous_ocr_review_files: set[str]) -> dict[str, Any]:
    filename = str(item.get("filename"))
    status = str(item.get("status"))
    was_phase37_ocr_review = filename in previous_ocr_review_files
    return {
        "filename": filename,
        "status": status,
        "entity_count": int(item.get("entity_count") or 0),
        "route_decision": item.get("route_decision"),
        "input_quality_score": item.get("input_quality_score"),
        "input_quality_band": item.get("input_quality_band"),
        "input_quality_warnings": list(item.get("input_quality_warnings") or []),
        "selected_extraction_engine": item.get("selected_engine"),
        "selected_extractor": item.get("selected_extractor"),
        "review_type": item.get("review_type"),
        "is_ocr_low_quality": bool(item.get("is_ocr_low_quality", False)),
        "downstream_classifier_status": item.get("downstream_classifier_status"),
        "downstream_classifier_reason": item.get("downstream_classifier_reason"),
        "classification_reason_codes": list(item.get("classification_reason_codes") or []),
        "empty_extraction_flag": bool(item.get("empty_extraction_flag", False)),
        "table_layout_warning": bool(item.get("table_layout_warning", False)),
        "ocr_status_mismatch": bool(item.get("ocr_status_mismatch", False)),
        "mismatch_type": item.get("mismatch_type"),
        "phase37_was_review_ocr_quality": was_phase37_ocr_review,
        "phase37_review_ocr_quality_improved": bool(was_phase37_ocr_review and status != "review_ocr_quality"),
        "why_reviewed": list(item.get("why_reviewed") or []),
    }


def _load_phase37_holdout_results() -> list[dict[str, Any]]:
    path = ROOT / "reports" / "holdout_validation" / "latest_holdout_validation.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return [item for item in payload.get("results", []) if isinstance(item, dict)]


def render_markdown(report: dict[str, Any]) -> str:
    baseline = report["phase37_holdout_baseline"]
    lines = [
        "# Phase 38 OCR/Layout Validation",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Input dir: `{report['source_input_dir']}`",
        "",
        "## Summary",
        "",
        f"- Total files: `{report['total_files']}`",
        f"- Accepted: `{report['accepted']}`",
        f"- Review: `{report['review']}`",
        f"- review_ocr_quality: `{report['review_ocr_quality']}`",
        f"- Empty: `{report['empty']}`",
        f"- OCR/Layout routes: `{report['ocr_layout_route_summary']}`",
        f"- OCR/Layout quality bands: `{report['ocr_layout_quality_summary']}`",
        "",
        "## Phase 37 Comparison",
        "",
        f"- Phase37 holdout total: `{baseline['total_files']}`",
        f"- Phase37 accepted: `{baseline['accepted']}`",
        f"- Phase37 review_ocr_quality: `{baseline['review_ocr_quality']}`",
        f"- Phase37 empty: `{baseline['empty']}`",
        f"- Phase37 review_ocr_quality improved: `{report['phase37_review_ocr_quality_improved_count']}`",
        f"- OCR status mismatches: `{report.get('ocr_status_mismatch_count', 0)}`",
        f"- OCR/Layout improved input quality: `{report['input_quality_improved']}`",
        f"- Safety regression: `{report['safety_regression']}`",
    ]
    if report.get("safety_regression_reason"):
        lines.append(f"- Safety regression reason: `{report['safety_regression_reason']}`")

    lines.extend([
        "",
        "## Per-file OCR/Layout Decisions",
        "",
        "| File | Status | Route | Quality | Score | Selected engine | Mismatch | Reason codes | Phase37 OCR review improved |",
        "| --- | --- | --- | --- | ---: | --- | --- | --- | --- |",
    ])
    for item in report.get("results", []):
        lines.append(
            "| "
            + " | ".join([
                _escape_md(item.get("filename")),
                _escape_md(item.get("status")),
                _escape_md(item.get("route_decision")),
                _escape_md(item.get("input_quality_band")),
                "" if item.get("input_quality_score") is None else str(item.get("input_quality_score")),
                _escape_md(item.get("selected_extraction_engine")),
                _escape_md(item.get("mismatch_type")) if item.get("ocr_status_mismatch") else "no",
                _escape_md(", ".join(item.get("classification_reason_codes") or [])),
                "yes" if item.get("phase37_review_ocr_quality_improved") else "no",
            ])
            + " |"
        )
    return "\n".join(lines) + "\n"


def _escape_md(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def main() -> int:
    report = run_phase38_validation()
    print("MedAI Phase 38 OCR/Layout validation complete.")
    print(f"total: {report['total_files']}")
    print(f"accepted: {report['accepted']}")
    print(f"review: {report['review']}")
    print(f"review_ocr_quality: {report['review_ocr_quality']}")
    print(f"empty: {report['empty']}")
    print(f"input_quality_improved: {report['input_quality_improved']}")
    print(f"safety_regression: {report['safety_regression']}")
    print(f"json_report: {PHASE38_JSON_REPORT}")
    print(f"markdown_report: {PHASE38_MD_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
