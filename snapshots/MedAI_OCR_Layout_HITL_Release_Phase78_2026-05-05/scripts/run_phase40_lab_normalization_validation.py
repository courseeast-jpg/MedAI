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
PHASE40_REPORT_DIR = ROOT / "reports" / "phase40_lab_normalization"
PHASE40_ARCHIVE_DIR = PHASE40_REPORT_DIR / "archive"
PHASE40_REVIEW_DIR = PHASE40_REPORT_DIR / "review"
PHASE40_ERROR_DIR = PHASE40_REPORT_DIR / "error"
PHASE40_JSON_REPORT = PHASE40_REPORT_DIR / "phase40_lab_normalization_report.json"
PHASE40_MD_REPORT = PHASE40_REPORT_DIR / "phase40_lab_normalization_report.md"

PHASE37_BASELINE = {"total_files": 8, "accepted": 2, "review_ocr_quality": 6, "empty": 0}
PHASE38_BASELINE = {"total_files": 8, "accepted": 2, "review": 6, "review_ocr_quality": 4, "empty": 0}
PHASE39_BASELINE = {
    "total_files": 8,
    "ocr_status_mismatches": 3,
    "review_ocr_quality": 4,
    "safety_regression": False,
}


def configure_phase40_paths() -> None:
    batch.REAL_VALIDATION_INPUT_DIR = HOLDOUT_INPUT_DIR
    batch.BATCH_REPORT_DIR = PHASE40_REPORT_DIR
    batch.BATCH_ARCHIVE_DIR = PHASE40_ARCHIVE_DIR
    batch.BATCH_REVIEW_DIR = PHASE40_REVIEW_DIR
    batch.BATCH_ERROR_DIR = PHASE40_ERROR_DIR
    batch.BATCH_JSON_REPORT = PHASE40_REPORT_DIR / "latest_phase40_batch_validation.json"
    batch.BATCH_MD_REPORT = PHASE40_REPORT_DIR / "latest_phase40_batch_validation.md"
    batch.REVIEW_AUDIT_JSON_REPORT = PHASE40_REPORT_DIR / "review_audit.json"
    batch.REVIEW_AUDIT_MD_REPORT = PHASE40_REPORT_DIR / "review_audit.md"


def run_phase40_validation() -> dict[str, Any]:
    configure_phase40_paths()
    PHASE40_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = batch.run_batch_validation()
    report = build_phase40_report(summary)
    PHASE40_JSON_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    PHASE40_MD_REPORT.write_text(render_markdown(report), encoding="utf-8")
    return report


def build_phase40_report(summary: dict[str, Any]) -> dict[str, Any]:
    results = [phase40_file_row(item) for item in summary.get("results", []) if isinstance(item, dict)]
    accepted = sum(1 for item in results if item["final_status_after_lab_normalization"] == "accepted")
    review = sum(1 for item in results if item["final_status_after_lab_normalization"] in {"review", "review_ocr_quality"})
    review_ocr_quality = sum(1 for item in results if item["final_status_after_lab_normalization"] == "review_ocr_quality")
    empty = sum(1 for item in results if item["empty_extraction_flag"])
    safety = safety_section(results)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "phase": "Phase 40 Lab Table Normalization & Coverage Recovery",
        "phase37_baseline": dict(PHASE37_BASELINE),
        "phase38_baseline": dict(PHASE38_BASELINE),
        "phase39_baseline": dict(PHASE39_BASELINE),
        "phase40_result": {
            "total_files": len(results),
            "accepted": accepted,
            "review": review,
            "review_ocr_quality": review_ocr_quality,
            "empty": empty,
            "review_ocr_quality_decreased_from_phase39": review_ocr_quality < PHASE39_BASELINE["review_ocr_quality"],
        },
        "safety": safety,
        "results": results,
        "raw_batch_report": str(batch.BATCH_JSON_REPORT),
        "raw_batch_markdown": str(batch.BATCH_MD_REPORT),
    }


def phase40_file_row(item: dict[str, Any]) -> dict[str, Any]:
    lab = item.get("lab_normalization") if isinstance(item.get("lab_normalization"), dict) else {}
    rows = lab.get("normalized_lab_rows") if isinstance(lab.get("normalized_lab_rows"), list) else []
    return {
        "filename": item.get("filename"),
        "ocr_layout_quality_band": item.get("input_quality_band"),
        "final_status_before_lab_normalization": item.get("final_status_before_lab_normalization", item.get("status")),
        "final_status_after_lab_normalization": item.get("final_status_after_lab_normalization", item.get("status")),
        "lab_table_detected": bool(lab.get("lab_table_detected", False)),
        "parsed_lab_row_count": len(rows),
        "lab_coverage_ratio": lab.get("lab_coverage_ratio", 0.0),
        "lab_coverage_band": lab.get("lab_coverage_band", "none"),
        "reason_codes": list(item.get("classification_reason_codes") or []),
        "original_reason_codes": list(item.get("original_classification_reason_codes") or []),
        "upgraded_from_review_ocr_quality_to_review": bool(item.get("lab_normalizer_changed_status", False)),
        "accepted_due_to_lab_normalizer": bool(item.get("accepted_due_to_lab_normalizer", False)),
        "empty_extraction_flag": bool(item.get("empty_extraction_flag", False)),
    }


def safety_section(results: list[dict[str, Any]]) -> dict[str, bool]:
    return {
        "false_accept_on_poor_ocr": any(
            item["final_status_after_lab_normalization"] == "accepted"
            and item["ocr_layout_quality_band"] in {"poor_ocr", "empty"}
            for item in results
        ),
        "accepted_due_to_lab_normalizer": any(item["accepted_due_to_lab_normalizer"] for item in results),
        "empty_extraction_leakage": any(
            item["final_status_after_lab_normalization"] == "accepted" and item["empty_extraction_flag"]
            for item in results
        ),
        "phase37_gate_bypassed": any(item["accepted_due_to_lab_normalizer"] for item in results),
    }


def render_markdown(report: dict[str, Any]) -> str:
    result = report["phase40_result"]
    safety = report["safety"]
    lines = [
        "# Phase 40 Lab Table Normalization",
        "",
        f"- Generated at: `{report['generated_at']}`",
        "",
        "## Baselines",
        "",
        f"- Phase37: `{report['phase37_baseline']}`",
        f"- Phase38: `{report['phase38_baseline']}`",
        f"- Phase39: `{report['phase39_baseline']}`",
        "",
        "## Phase40 Result",
        "",
        f"- Total files: `{result['total_files']}`",
        f"- Accepted: `{result['accepted']}`",
        f"- Review: `{result['review']}`",
        f"- review_ocr_quality: `{result['review_ocr_quality']}`",
        f"- Empty: `{result['empty']}`",
        f"- review_ocr_quality decreased from Phase39: `{result['review_ocr_quality_decreased_from_phase39']}`",
        "",
        "## Safety",
        "",
        f"- false_accept_on_poor_ocr: `{safety['false_accept_on_poor_ocr']}`",
        f"- accepted_due_to_lab_normalizer: `{safety['accepted_due_to_lab_normalizer']}`",
        f"- empty_extraction_leakage: `{safety['empty_extraction_leakage']}`",
        f"- phase37_gate_bypassed: `{safety['phase37_gate_bypassed']}`",
        "",
        "## Per-file Results",
        "",
        "| File | OCR band | Before | After | Lab table | Rows | Coverage | Band | Recovered | Reason codes |",
        "| --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for item in report["results"]:
        lines.append(
            "| "
            + " | ".join([
                _escape_md(item.get("filename")),
                _escape_md(item.get("ocr_layout_quality_band")),
                _escape_md(item.get("final_status_before_lab_normalization")),
                _escape_md(item.get("final_status_after_lab_normalization")),
                "yes" if item.get("lab_table_detected") else "no",
                str(item.get("parsed_lab_row_count", 0)),
                str(item.get("lab_coverage_ratio", 0.0)),
                _escape_md(item.get("lab_coverage_band")),
                "yes" if item.get("upgraded_from_review_ocr_quality_to_review") else "no",
                _escape_md(", ".join(item.get("reason_codes") or [])),
            ])
            + " |"
        )
    return "\n".join(lines) + "\n"


def _escape_md(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def main() -> int:
    report = run_phase40_validation()
    result = report["phase40_result"]
    safety = report["safety"]
    print("MedAI Phase 40 lab normalization validation complete.")
    print(f"total: {result['total_files']}")
    print(f"accepted: {result['accepted']}")
    print(f"review: {result['review']}")
    print(f"review_ocr_quality: {result['review_ocr_quality']}")
    print(f"empty: {result['empty']}")
    print(f"review_ocr_quality_decreased: {result['review_ocr_quality_decreased_from_phase39']}")
    print(f"safety: {safety}")
    print(f"json_report: {PHASE40_JSON_REPORT}")
    print(f"markdown_report: {PHASE40_MD_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
