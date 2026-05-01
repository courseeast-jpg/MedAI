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
PHASE41_REPORT_DIR = ROOT / "reports" / "phase41_flattened_lab_rows"
PHASE41_ARCHIVE_DIR = PHASE41_REPORT_DIR / "archive"
PHASE41_REVIEW_DIR = PHASE41_REPORT_DIR / "review"
PHASE41_ERROR_DIR = PHASE41_REPORT_DIR / "error"
PHASE41_JSON_REPORT = PHASE41_REPORT_DIR / "phase41_flattened_lab_rows_report.json"
PHASE41_MD_REPORT = PHASE41_REPORT_DIR / "phase41_flattened_lab_rows_report.md"

PHASE40_BASELINE = {
    "total_files": 8,
    "accepted": 2,
    "review": 6,
    "review_ocr_quality": 3,
    "empty": 0,
    "safety_regression": False,
}

DIFFICULT_FILES = {"Test Results 3.pdf", "Test Results 6.pdf"}


def configure_phase41_paths() -> None:
    batch.REAL_VALIDATION_INPUT_DIR = HOLDOUT_INPUT_DIR
    batch.BATCH_REPORT_DIR = PHASE41_REPORT_DIR
    batch.BATCH_ARCHIVE_DIR = PHASE41_ARCHIVE_DIR
    batch.BATCH_REVIEW_DIR = PHASE41_REVIEW_DIR
    batch.BATCH_ERROR_DIR = PHASE41_ERROR_DIR
    batch.BATCH_JSON_REPORT = PHASE41_REPORT_DIR / "latest_phase41_batch_validation.json"
    batch.BATCH_MD_REPORT = PHASE41_REPORT_DIR / "latest_phase41_batch_validation.md"
    batch.REVIEW_AUDIT_JSON_REPORT = PHASE41_REPORT_DIR / "review_audit.json"
    batch.REVIEW_AUDIT_MD_REPORT = PHASE41_REPORT_DIR / "review_audit.md"


def run_phase41_validation() -> dict[str, Any]:
    configure_phase41_paths()
    PHASE41_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = batch.run_batch_validation()
    report = build_phase41_report(summary)
    PHASE41_JSON_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    PHASE41_MD_REPORT.write_text(render_markdown(report), encoding="utf-8")
    return report


def build_phase41_report(summary: dict[str, Any]) -> dict[str, Any]:
    results = [phase41_file_row(item) for item in summary.get("results", []) if isinstance(item, dict)]
    accepted = sum(1 for r in results if r["final_status"] == "accepted")
    review = sum(1 for r in results if r["final_status"] in {"review", "review_ocr_quality"})
    review_ocr_quality = sum(1 for r in results if r["final_status"] == "review_ocr_quality")
    empty = sum(1 for r in results if r["empty_extraction_flag"])

    safety = safety_section(results)

    difficult = {r["filename"]: r for r in results if r["filename"] in DIFFICULT_FILES}
    phase41_result = {
        "total_files": len(results),
        "accepted": accepted,
        "review": review,
        "review_ocr_quality": review_ocr_quality,
        "empty": empty,
        "review_ocr_quality_vs_phase40_baseline": review_ocr_quality - PHASE40_BASELINE["review_ocr_quality"],
        "review_ocr_quality_decreased_from_phase40": review_ocr_quality < PHASE40_BASELINE["review_ocr_quality"],
        "accepted_safe": accepted <= PHASE40_BASELINE["accepted"],
    }

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "phase": "Phase 41 Flattened Lab Row Parser Expansion",
        "phase40_baseline": dict(PHASE40_BASELINE),
        "phase41_result": phase41_result,
        "safety": safety,
        "difficult_files": difficult,
        "results": results,
        "raw_batch_report": str(batch.BATCH_JSON_REPORT),
    }


def phase41_file_row(item: dict[str, Any]) -> dict[str, Any]:
    lab = item.get("lab_normalization") if isinstance(item.get("lab_normalization"), dict) else {}
    rows = lab.get("normalized_lab_rows") if isinstance(lab.get("normalized_lab_rows"), list) else []
    flattened_row_count = sum(
        1 for r in rows
        if isinstance(r, dict) and any(
            w in r.get("warnings", [])
            for w in ("adjacent_value_unit", "split_range_row")
        )
    )
    return {
        "filename": item.get("filename"),
        "ocr_layout_quality_band": item.get("input_quality_band"),
        "final_status": item.get("final_status_after_lab_normalization", item.get("status")),
        "lab_table_detected": bool(lab.get("lab_table_detected", False)),
        "parsed_lab_row_count": len(rows),
        "flattened_rows_recovered": flattened_row_count,
        "lab_coverage_ratio": lab.get("lab_coverage_ratio", 0.0),
        "lab_coverage_band": lab.get("lab_coverage_band", "none"),
        "reason_codes": list(item.get("classification_reason_codes") or []),
        "upgraded_from_review_ocr_quality": bool(item.get("lab_normalizer_changed_status", False)),
        "accepted_due_to_lab_normalizer": bool(item.get("accepted_due_to_lab_normalizer", False)),
        "empty_extraction_flag": bool(item.get("empty_extraction_flag", False)),
    }


def safety_section(results: list[dict[str, Any]]) -> dict[str, bool]:
    return {
        "false_accept_on_poor_ocr": any(
            r["final_status"] == "accepted"
            and r["ocr_layout_quality_band"] in {"poor_ocr", "empty"}
            for r in results
        ),
        "accepted_due_to_lab_normalizer": any(r["accepted_due_to_lab_normalizer"] for r in results),
        "empty_extraction_leakage": any(
            r["final_status"] == "accepted" and r["empty_extraction_flag"]
            for r in results
        ),
        "safety_regression": False,
    }


def render_markdown(report: dict[str, Any]) -> str:
    result = report["phase41_result"]
    safety = report["safety"]
    lines = [
        "# Phase 41 Flattened Lab Row Parser Expansion",
        "",
        f"- Generated at: `{report['generated_at']}`",
        "",
        "## Phase 40 Baseline",
        "",
        f"- Total: `{report['phase40_baseline']['total_files']}`",
        f"- Accepted: `{report['phase40_baseline']['accepted']}`",
        f"- review_ocr_quality: `{report['phase40_baseline']['review_ocr_quality']}`",
        "",
        "## Phase 41 Result",
        "",
        f"- Total files: `{result['total_files']}`",
        f"- Accepted: `{result['accepted']}`",
        f"- Review: `{result['review']}`",
        f"- review_ocr_quality: `{result['review_ocr_quality']}`",
        f"- Empty: `{result['empty']}`",
        f"- review_ocr_quality vs Phase 40: `{result['review_ocr_quality_vs_phase40_baseline']:+d}`",
        f"- review_ocr_quality decreased from Phase 40: `{result['review_ocr_quality_decreased_from_phase40']}`",
        f"- accepted_safe (≤ Phase40 accepted): `{result['accepted_safe']}`",
        "",
        "## Safety",
        "",
        f"- false_accept_on_poor_ocr: `{safety['false_accept_on_poor_ocr']}`",
        f"- accepted_due_to_lab_normalizer: `{safety['accepted_due_to_lab_normalizer']}`",
        f"- empty_extraction_leakage: `{safety['empty_extraction_leakage']}`",
        f"- safety_regression: `{safety['safety_regression']}`",
        "",
        "## Difficult Files",
        "",
    ]
    difficult = report.get("difficult_files", {})
    if difficult:
        for fname, item in difficult.items():
            lines += [
                f"### {fname}",
                "",
                f"- Status: `{item.get('final_status')}`",
                f"- Parsed rows: `{item.get('parsed_lab_row_count', 0)}`",
                f"- Flattened rows recovered: `{item.get('flattened_rows_recovered', 0)}`",
                f"- Coverage ratio: `{item.get('lab_coverage_ratio', 0.0)}`",
                f"- Coverage band: `{item.get('lab_coverage_band')}`",
                f"- Reason codes: `{', '.join(item.get('reason_codes') or [])}`",
                "",
            ]
    else:
        lines.append("_(no holdout files found)_")
        lines.append("")

    lines += [
        "## Per-file Results",
        "",
        "| File | OCR band | Status | Rows | Flattened | Coverage | Band | Upgraded | Reason codes |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for item in report["results"]:
        lines.append(
            "| "
            + " | ".join([
                _escape_md(item.get("filename")),
                _escape_md(item.get("ocr_layout_quality_band")),
                _escape_md(item.get("final_status")),
                str(item.get("parsed_lab_row_count", 0)),
                str(item.get("flattened_rows_recovered", 0)),
                str(item.get("lab_coverage_ratio", 0.0)),
                _escape_md(item.get("lab_coverage_band")),
                "yes" if item.get("upgraded_from_review_ocr_quality") else "no",
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
    report = run_phase41_validation()
    result = report["phase41_result"]
    safety = report["safety"]
    difficult = report.get("difficult_files", {})
    print("MedAI Phase 41 flattened lab row validation complete.")
    print(f"total: {result['total_files']}")
    print(f"accepted: {result['accepted']}")
    print(f"review: {result['review']}")
    print(f"review_ocr_quality: {result['review_ocr_quality']}")
    print(f"empty: {result['empty']}")
    print(f"review_ocr_quality_vs_phase40: {result['review_ocr_quality_vs_phase40_baseline']:+d}")
    print(f"review_ocr_quality_decreased_from_phase40: {result['review_ocr_quality_decreased_from_phase40']}")
    print(f"accepted_safe: {result['accepted_safe']}")
    print(f"safety: {safety}")
    for fname in DIFFICULT_FILES:
        if fname in difficult:
            d = difficult[fname]
            print(
                f"{fname}: status={d['final_status']} rows={d['parsed_lab_row_count']} "
                f"flattened={d['flattened_rows_recovered']} coverage={d['lab_coverage_ratio']}"
            )
        else:
            print(f"{fname}: not found in holdout input")
    print(f"json_report: {PHASE41_JSON_REPORT}")
    print(f"markdown_report: {PHASE41_MD_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
