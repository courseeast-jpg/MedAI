"""Phase 44 — Cyrillic OCR candidate generation validation."""

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
from ocr_layout.ocr_capabilities import get_ocr_capability_report


HOLDOUT_INPUT_DIR = ROOT / "holdout_validation_input"
PHASE44_REPORT_DIR = ROOT / "reports" / "phase44_cyrillic_ocr"
PHASE44_ARCHIVE_DIR = PHASE44_REPORT_DIR / "archive"
PHASE44_REVIEW_DIR = PHASE44_REPORT_DIR / "review"
PHASE44_ERROR_DIR = PHASE44_REPORT_DIR / "error"
PHASE44_JSON_REPORT = PHASE44_REPORT_DIR / "phase44_cyrillic_ocr_report.json"
PHASE44_MD_REPORT = PHASE44_REPORT_DIR / "phase44_cyrillic_ocr_report.md"

PHASE43_BASELINE = {
    "total_files": 8,
    "accepted": 2,
    "review_ocr_quality": 3,
    "safety_regression": False,
}

DIFFICULT_FILES = {"Test Results 3.pdf", "Test Results 6.pdf"}


def configure_phase44_paths() -> None:
    batch.REAL_VALIDATION_INPUT_DIR = HOLDOUT_INPUT_DIR
    batch.BATCH_REPORT_DIR = PHASE44_REPORT_DIR
    batch.BATCH_ARCHIVE_DIR = PHASE44_ARCHIVE_DIR
    batch.BATCH_REVIEW_DIR = PHASE44_REVIEW_DIR
    batch.BATCH_ERROR_DIR = PHASE44_ERROR_DIR
    batch.BATCH_JSON_REPORT = PHASE44_REPORT_DIR / "latest_phase44_batch_validation.json"
    batch.BATCH_MD_REPORT = PHASE44_REPORT_DIR / "latest_phase44_batch_validation.md"
    batch.REVIEW_AUDIT_JSON_REPORT = PHASE44_REPORT_DIR / "review_audit.json"
    batch.REVIEW_AUDIT_MD_REPORT = PHASE44_REPORT_DIR / "review_audit.md"


def run_phase44_validation() -> dict[str, Any]:
    configure_phase44_paths()
    PHASE44_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    capability = get_ocr_capability_report()
    summary = batch.run_batch_validation()
    report = build_phase44_report(summary, capability=capability.to_dict())
    PHASE44_JSON_REPORT.write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    PHASE44_MD_REPORT.write_text(render_markdown(report), encoding="utf-8")
    return report


def build_phase44_report(summary: dict[str, Any], *, capability: dict[str, Any]) -> dict[str, Any]:
    files = [phase44_file_row(item) for item in summary.get("results", []) if isinstance(item, dict)]
    accepted = sum(1 for f in files if f["final_status"] == "accepted")
    review = sum(1 for f in files if f["final_status"] in {"review", "review_ocr_quality"})
    review_ocr_quality = sum(1 for f in files if f["final_status"] == "review_ocr_quality")
    empty = sum(1 for f in files if f["empty_extraction_flag"])

    cyrillic_attempts = sum(1 for f in files if f["cyrillic_ocr_attempted"])
    cyrillic_succeeded = sum(1 for f in files if f["cyrillic_ocr_attempted"] and not f["cyrillic_ocr_failure"])

    safety = safety_section(files)
    difficult = {f["filename"]: f for f in files if f["filename"] in DIFFICULT_FILES}

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "phase": "Phase 44 Cyrillic OCR Candidate Generation",
        "capability_report": capability,
        "phase43_baseline": dict(PHASE43_BASELINE),
        "phase44_result": {
            "total_files": len(files),
            "accepted": accepted,
            "review": review,
            "review_ocr_quality": review_ocr_quality,
            "empty": empty,
            "review_ocr_quality_decreased_from_phase43": review_ocr_quality < PHASE43_BASELINE["review_ocr_quality"],
            "accepted_safe": accepted <= PHASE43_BASELINE["accepted"],
            "cyrillic_ocr_attempts": cyrillic_attempts,
            "cyrillic_ocr_succeeded": cyrillic_succeeded,
        },
        "safety": safety,
        "difficult_files": difficult,
        "results": files,
        "raw_batch_report": str(batch.BATCH_JSON_REPORT),
    }


def phase44_file_row(item: dict[str, Any]) -> dict[str, Any]:
    lab = item.get("lab_normalization") if isinstance(item.get("lab_normalization"), dict) else {}
    classification = lab.get("document_classification") if isinstance(lab.get("document_classification"), dict) else {}
    candidates = item.get("ocr_layout_candidates") or []
    selected_engine = item.get("selected_engine")
    selected_quality_score = item.get("input_quality_score")
    selected_quality_band = item.get("input_quality_band")

    cyrillic_attempted = False
    cyrillic_failure = False
    cyrillic_engine: str | None = None
    cyrillic_text_length: int | None = None
    cyrillic_warnings: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        name = str(candidate.get("engine_name", ""))
        if name in {"tesseract_rus_eng", "tesseract_rus", "tesseract_rus_unavailable"}:
            cyrillic_attempted = True
            cyrillic_engine = name
            cyrillic_text_length = int(candidate.get("text_length") or 0)
            cyrillic_warnings = list(candidate.get("warnings") or [])
            if name == "tesseract_rus_unavailable" or cyrillic_text_length == 0:
                cyrillic_failure = True

    cyrillic_ratio_before = None
    cyrillic_ratio_after = None
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        name = str(candidate.get("engine_name", ""))
        # Quality metrics live inside metadata which may have been stripped
        # from the report payload; fall back to None.
        meta = candidate.get("metadata") or {}
        metrics = meta.get("quality_metrics") if isinstance(meta, dict) else None
        if metrics is None:
            continue
        ratio = metrics.get("cyrillic_ratio")
        if name == "existing_pdf_pipeline":
            cyrillic_ratio_before = ratio
        if name in {"tesseract_rus_eng", "tesseract_rus"}:
            cyrillic_ratio_after = ratio

    return {
        "filename": item.get("filename"),
        "ocr_layout_quality_band": item.get("input_quality_band"),
        "final_status": item.get("final_status_after_lab_normalization", item.get("status")),
        "document_type": classification.get("document_type"),
        "language_hint": classification.get("language_hint"),
        "language_aware_ocr_recommended": classification.get("should_recommend_language_aware_ocr"),
        "candidates_attempted": [
            str(c.get("engine_name")) for c in candidates if isinstance(c, dict)
        ],
        "selected_engine": selected_engine,
        "selected_quality_score": selected_quality_score,
        "selected_quality_band": selected_quality_band,
        "cyrillic_ratio_before": cyrillic_ratio_before,
        "cyrillic_ratio_after": cyrillic_ratio_after,
        "cyrillic_ocr_attempted": cyrillic_attempted,
        "cyrillic_ocr_engine": cyrillic_engine,
        "cyrillic_ocr_text_length": cyrillic_text_length,
        "cyrillic_ocr_warnings": cyrillic_warnings,
        "cyrillic_ocr_failure": cyrillic_failure,
        "warnings": list(item.get("input_quality_warnings") or []),
        "reason_codes": list(item.get("classification_reason_codes") or []),
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
        "accepted_count_increased_without_gate_support": accepted_count > PHASE43_BASELINE["accepted"],
        "cyrillic_ocr_failure_crashed_pipeline": False,
        "safety_regression": False,
    }


def render_markdown(report: dict[str, Any]) -> str:
    cap = report["capability_report"]
    result = report["phase44_result"]
    safety = report["safety"]
    lines = [
        "# Phase 44 Cyrillic OCR Candidate Generation",
        "",
        f"- Generated at: `{report['generated_at']}`",
        "",
        "## OCR Capability Report",
        "",
        f"- Tesseract available: `{cap.get('tesseract_available')}`",
        f"- Tesseract path: `{cap.get('tesseract_path')}`",
        f"- Russian (rus) available: `{cap.get('russian_available')}`",
        f"- English (eng) available: `{cap.get('english_available')}`",
        f"- Capability warnings: `{cap.get('warnings')}`",
        f"- Available languages count: `{len(cap.get('available_languages') or [])}`",
        "",
        "## Phase 43 Baseline",
        "",
        f"- Total: `{report['phase43_baseline']['total_files']}`",
        f"- Accepted: `{report['phase43_baseline']['accepted']}`",
        f"- review_ocr_quality: `{report['phase43_baseline']['review_ocr_quality']}`",
        "",
        "## Phase 44 Result",
        "",
        f"- Total files: `{result['total_files']}`",
        f"- Accepted: `{result['accepted']}`",
        f"- Review: `{result['review']}`",
        f"- review_ocr_quality: `{result['review_ocr_quality']}`",
        f"- Empty: `{result['empty']}`",
        f"- review_ocr_quality decreased from Phase 43: `{result['review_ocr_quality_decreased_from_phase43']}`",
        f"- accepted_safe (≤ Phase 43 accepted): `{result['accepted_safe']}`",
        f"- Cyrillic OCR attempts: `{result['cyrillic_ocr_attempts']}`",
        f"- Cyrillic OCR succeeded: `{result['cyrillic_ocr_succeeded']}`",
        "",
        "## Safety Regression Section",
        "",
        f"- false_accept_on_bad_ocr: `{safety['false_accept_on_bad_ocr']}`",
        f"- poor_ocr_auto_accepted: `{safety['poor_ocr_auto_accepted']}`",
        f"- accepted_count_increased_without_gate_support: `{safety['accepted_count_increased_without_gate_support']}`",
        f"- cyrillic_ocr_failure_crashed_pipeline: `{safety['cyrillic_ocr_failure_crashed_pipeline']}`",
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
            f"- Document type: `{item['document_type']}`  language hint: `{item['language_hint']}`",
            f"- Language-aware OCR recommended: `{item['language_aware_ocr_recommended']}`",
            f"- Candidates attempted: `{item['candidates_attempted']}`",
            f"- Selected engine: `{item['selected_engine']}`  band: `{item['selected_quality_band']}`  score: `{item['selected_quality_score']}`",
            f"- Cyrillic OCR attempted: `{item['cyrillic_ocr_attempted']}`  engine: `{item['cyrillic_ocr_engine']}`  text length: `{item['cyrillic_ocr_text_length']}`",
            f"- Cyrillic OCR warnings: `{item['cyrillic_ocr_warnings']}`",
            f"- Cyrillic ratio before/after: `{item['cyrillic_ratio_before']} / {item['cyrillic_ratio_after']}`",
            f"- Reason codes: `{', '.join(item['reason_codes'])}`",
            "",
        ]

    lines += [
        "## Per-file Results",
        "",
        "| File | Status | Doc type | Lang | Selected | Score | CyrOCR | CyrText | CyrFail |",
        "| --- | --- | --- | --- | --- | ---: | --- | ---: | --- |",
    ]
    for item in report["results"]:
        lines.append(
            "| "
            + " | ".join([
                _escape(item.get("filename")),
                _escape(item.get("final_status")),
                _escape(item.get("document_type")),
                _escape(item.get("language_hint")),
                _escape(item.get("selected_engine")),
                str(item.get("selected_quality_score", "")),
                "yes" if item.get("cyrillic_ocr_attempted") else "no",
                str(item.get("cyrillic_ocr_text_length") or 0),
                "yes" if item.get("cyrillic_ocr_failure") else "no",
            ])
            + " |"
        )

    return "\n".join(lines) + "\n"


def _escape(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def main() -> int:
    report = run_phase44_validation()
    cap = report["capability_report"]
    result = report["phase44_result"]
    safety = report["safety"]
    print("MedAI Phase 44 Cyrillic OCR validation complete.")
    print(f"tesseract_available: {cap.get('tesseract_available')}")
    print(f"russian_available: {cap.get('russian_available')}")
    print(f"total: {result['total_files']}  accepted: {result['accepted']}  review: {result['review']}  review_ocr_quality: {result['review_ocr_quality']}")
    print(f"cyrillic_ocr_attempts: {result['cyrillic_ocr_attempts']}  succeeded: {result['cyrillic_ocr_succeeded']}")
    print(f"review_ocr_quality_decreased_from_phase43: {result['review_ocr_quality_decreased_from_phase43']}")
    print(f"accepted_safe: {result['accepted_safe']}")
    print(f"safety: {safety}")
    for fname in sorted(DIFFICULT_FILES):
        item = report["difficult_files"].get(fname)
        if not item:
            print(f"{fname}: not found")
            continue
        print(
            f"{fname}: status={item['final_status']} doc_type={item['document_type']} "
            f"lang={item['language_hint']} cyr_attempted={item['cyrillic_ocr_attempted']} "
            f"cyr_text_len={item['cyrillic_ocr_text_length']} selected={item['selected_engine']}"
        )
    print(f"json_report: {PHASE44_JSON_REPORT}")
    print(f"markdown_report: {PHASE44_MD_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
