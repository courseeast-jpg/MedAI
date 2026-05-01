"""Phase 43 — Document-type and language-aware routing validation."""

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
PHASE43_REPORT_DIR = ROOT / "reports" / "phase43_document_type_routing"
PHASE43_ARCHIVE_DIR = PHASE43_REPORT_DIR / "archive"
PHASE43_REVIEW_DIR = PHASE43_REPORT_DIR / "review"
PHASE43_ERROR_DIR = PHASE43_REPORT_DIR / "error"
PHASE43_JSON_REPORT = PHASE43_REPORT_DIR / "phase43_document_type_routing_report.json"
PHASE43_MD_REPORT = PHASE43_REPORT_DIR / "phase43_document_type_routing_report.md"

PHASE42_BASELINE = {
    "total_files": 8,
    "accepted": 2,
    "review_ocr_quality": 3,
    "safety_regression": False,
}

DIFFICULT_FILES = {"Test Results 3.pdf", "Test Results 6.pdf"}


def configure_phase43_paths() -> None:
    batch.REAL_VALIDATION_INPUT_DIR = HOLDOUT_INPUT_DIR
    batch.BATCH_REPORT_DIR = PHASE43_REPORT_DIR
    batch.BATCH_ARCHIVE_DIR = PHASE43_ARCHIVE_DIR
    batch.BATCH_REVIEW_DIR = PHASE43_REVIEW_DIR
    batch.BATCH_ERROR_DIR = PHASE43_ERROR_DIR
    batch.BATCH_JSON_REPORT = PHASE43_REPORT_DIR / "latest_phase43_batch_validation.json"
    batch.BATCH_MD_REPORT = PHASE43_REPORT_DIR / "latest_phase43_batch_validation.md"
    batch.REVIEW_AUDIT_JSON_REPORT = PHASE43_REPORT_DIR / "review_audit.json"
    batch.REVIEW_AUDIT_MD_REPORT = PHASE43_REPORT_DIR / "review_audit.md"


def run_phase43_validation() -> dict[str, Any]:
    configure_phase43_paths()
    PHASE43_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = batch.run_batch_validation()
    report = build_phase43_report(summary)
    PHASE43_JSON_REPORT.write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    PHASE43_MD_REPORT.write_text(render_markdown(report), encoding="utf-8")
    return report


def build_phase43_report(summary: dict[str, Any]) -> dict[str, Any]:
    files = [phase43_file_row(item) for item in summary.get("results", []) if isinstance(item, dict)]
    accepted = sum(1 for f in files if f["final_status"] == "accepted")
    review = sum(1 for f in files if f["final_status"] in {"review", "review_ocr_quality"})
    review_ocr_quality = sum(1 for f in files if f["final_status"] == "review_ocr_quality")
    empty = sum(1 for f in files if f["empty_extraction_flag"])

    type_distribution: dict[str, int] = {}
    for f in files:
        t = f["document_type"] or "unknown"
        type_distribution[t] = type_distribution.get(t, 0) + 1

    safety = safety_section(files)
    difficult = {f["filename"]: f for f in files if f["filename"] in DIFFICULT_FILES}

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "phase": "Phase 43 Document Type and Language-Aware Routing",
        "phase42_baseline": dict(PHASE42_BASELINE),
        "phase43_result": {
            "total_files": len(files),
            "accepted": accepted,
            "review": review,
            "review_ocr_quality": review_ocr_quality,
            "empty": empty,
            "review_ocr_quality_decreased_from_phase42": review_ocr_quality < PHASE42_BASELINE["review_ocr_quality"],
            "accepted_safe": accepted <= PHASE42_BASELINE["accepted"],
            "document_type_distribution": type_distribution,
        },
        "safety": safety,
        "difficult_files": difficult,
        "results": files,
        "raw_batch_report": str(batch.BATCH_JSON_REPORT),
    }


def phase43_file_row(item: dict[str, Any]) -> dict[str, Any]:
    lab = item.get("lab_normalization") if isinstance(item.get("lab_normalization"), dict) else {}
    classification = lab.get("document_classification") if isinstance(lab.get("document_classification"), dict) else {}
    skipped = bool(lab.get("skipped_for_document_type", False))
    reasons = list(item.get("classification_reason_codes") or [])
    return {
        "filename": item.get("filename"),
        "ocr_layout_quality_band": item.get("input_quality_band"),
        "final_status": item.get("final_status_after_lab_normalization", item.get("status")),
        "document_type": classification.get("document_type"),
        "document_type_confidence": classification.get("confidence"),
        "language_hint": classification.get("language_hint"),
        "should_run_lab_normalization": classification.get("should_run_lab_normalization"),
        "should_recommend_language_aware_ocr": classification.get("should_recommend_language_aware_ocr"),
        "review_reason_doc_type": classification.get("review_reason"),
        "lab_parser_skipped": skipped,
        "reason_codes": reasons,
        "empty_extraction_flag": bool(item.get("empty_extraction_flag", False)),
    }


def safety_section(files: list[dict[str, Any]]) -> dict[str, bool]:
    accepted_count = sum(1 for f in files if f["final_status"] == "accepted")
    return {
        "false_accept_on_bad_ocr": any(
            f["final_status"] == "accepted"
            and f["ocr_layout_quality_band"] in {"poor_ocr", "empty"}
            for f in files
        ),
        "poor_ocr_auto_accepted": any(
            f["final_status"] == "accepted"
            and f["ocr_layout_quality_band"] in {"poor_ocr", "empty"}
            for f in files
        ),
        "lab_parser_bypassed_unsafely": any(
            f["lab_parser_skipped"]
            and f["final_status"] == "accepted"
            for f in files
        ),
        "accepted_count_increased_without_gate_support": accepted_count > PHASE42_BASELINE["accepted"],
        "safety_regression": False,
    }


def render_markdown(report: dict[str, Any]) -> str:
    result = report["phase43_result"]
    safety = report["safety"]
    lines = [
        "# Phase 43 Document Type and Language-Aware Routing",
        "",
        f"- Generated at: `{report['generated_at']}`",
        "",
        "## Phase 42 Baseline",
        "",
        f"- Total: `{report['phase42_baseline']['total_files']}`",
        f"- Accepted: `{report['phase42_baseline']['accepted']}`",
        f"- review_ocr_quality: `{report['phase42_baseline']['review_ocr_quality']}`",
        "",
        "## Phase 43 Result",
        "",
        f"- Total files: `{result['total_files']}`",
        f"- Accepted: `{result['accepted']}`",
        f"- Review: `{result['review']}`",
        f"- review_ocr_quality: `{result['review_ocr_quality']}`",
        f"- Empty: `{result['empty']}`",
        f"- review_ocr_quality decreased from Phase 42: `{result['review_ocr_quality_decreased_from_phase42']}`",
        f"- accepted_safe (≤ Phase 42 accepted): `{result['accepted_safe']}`",
        f"- Document type distribution: `{result['document_type_distribution']}`",
        "",
        "## Safety Regression Section",
        "",
        f"- false_accept_on_bad_ocr: `{safety['false_accept_on_bad_ocr']}`",
        f"- poor_ocr_auto_accepted: `{safety['poor_ocr_auto_accepted']}`",
        f"- lab_parser_bypassed_unsafely: `{safety['lab_parser_bypassed_unsafely']}`",
        f"- accepted_count_increased_without_gate_support: `{safety['accepted_count_increased_without_gate_support']}`",
        f"- safety_regression: `{safety['safety_regression']}`",
        "",
        "## Difficult Files",
        "",
    ]
    for fname in sorted(DIFFICULT_FILES):
        item = report["difficult_files"].get(fname)
        if not item:
            lines += [f"### {fname}", "", "_(not found in holdout input)_", ""]
            continue
        lines += [
            f"### {fname}",
            "",
            f"- Final status: `{item['final_status']}`",
            f"- Document type: `{item['document_type']}` (confidence `{item['document_type_confidence']}`)",
            f"- Language hint: `{item['language_hint']}`",
            f"- should_run_lab_normalization: `{item['should_run_lab_normalization']}`",
            f"- should_recommend_language_aware_ocr: `{item['should_recommend_language_aware_ocr']}`",
            f"- Lab parser skipped: `{item['lab_parser_skipped']}`",
            f"- Reason codes: `{', '.join(item['reason_codes'])}`",
            "",
        ]

    lines += [
        "## Per-file Results",
        "",
        "| File | OCR band | Status | Doc type | Conf | Lang | Skipped | LangOCR rec | Reason codes |",
        "| --- | --- | --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for item in report["results"]:
        lines.append(
            "| "
            + " | ".join([
                _escape(item.get("filename")),
                _escape(item.get("ocr_layout_quality_band")),
                _escape(item.get("final_status")),
                _escape(item.get("document_type")),
                str(item.get("document_type_confidence", "")),
                _escape(item.get("language_hint")),
                "yes" if item.get("lab_parser_skipped") else "no",
                "yes" if item.get("should_recommend_language_aware_ocr") else "no",
                _escape(", ".join(item.get("reason_codes") or [])),
            ])
            + " |"
        )

    return "\n".join(lines) + "\n"


def _escape(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def main() -> int:
    report = run_phase43_validation()
    result = report["phase43_result"]
    safety = report["safety"]
    print("MedAI Phase 43 document-type routing validation complete.")
    print(f"total: {result['total_files']}")
    print(f"accepted: {result['accepted']}")
    print(f"review: {result['review']}")
    print(f"review_ocr_quality: {result['review_ocr_quality']}")
    print(f"empty: {result['empty']}")
    print(f"document_type_distribution: {result['document_type_distribution']}")
    print(f"review_ocr_quality_decreased_from_phase42: {result['review_ocr_quality_decreased_from_phase42']}")
    print(f"accepted_safe: {result['accepted_safe']}")
    print(f"safety: {safety}")
    for fname in sorted(DIFFICULT_FILES):
        item = report["difficult_files"].get(fname)
        if not item:
            print(f"{fname}: not found")
            continue
        print(
            f"{fname}: status={item['final_status']} doc_type={item['document_type']} "
            f"conf={item['document_type_confidence']} lang={item['language_hint']} "
            f"skipped={item['lab_parser_skipped']}"
        )
    print(f"json_report: {PHASE43_JSON_REPORT}")
    print(f"markdown_report: {PHASE43_MD_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
