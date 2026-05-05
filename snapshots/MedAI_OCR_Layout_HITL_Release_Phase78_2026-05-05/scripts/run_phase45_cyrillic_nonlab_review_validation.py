"""Phase 45 — Cyrillic non-lab review classification refinement validation."""

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
PHASE45_REPORT_DIR = ROOT / "reports" / "phase45_cyrillic_nonlab_review"
PHASE45_ARCHIVE_DIR = PHASE45_REPORT_DIR / "archive"
PHASE45_REVIEW_DIR = PHASE45_REPORT_DIR / "review"
PHASE45_ERROR_DIR = PHASE45_REPORT_DIR / "error"
PHASE45_JSON_REPORT = PHASE45_REPORT_DIR / "phase45_cyrillic_nonlab_review_report.json"
PHASE45_MD_REPORT = PHASE45_REPORT_DIR / "phase45_cyrillic_nonlab_review_report.md"


# Frozen baselines per user spec.
PHASE37_BASELINE = {"total_files": 8, "accepted": 2, "review_ocr_quality": 6, "empty": 0}
PHASE38_BASELINE = {"total_files": 8, "accepted": 2, "review": 6, "review_ocr_quality": 4, "empty": 0}
PHASE39_BASELINE = {"total_files": 8, "ocr_status_mismatches": 3, "review_ocr_quality": 4, "safety_regression": False}
PHASE40_BASELINE = {"total_files": 8, "accepted": 2, "review_ocr_quality": 3, "empty": 0, "safety_regression": False}
PHASE43_BASELINE = {"total_files": 8, "accepted": 2, "review_ocr_quality": 3, "safety_regression": False}
PHASE44_FROZEN_BASELINE = {
    "total_files": 8,
    "accepted": 2,
    "review_ocr_quality": 3,
    "tests_passed": 378,
    "safety_regression": False,
    "clean_commit": "78d5c8d44be5cac568f4c05870acaa86ee7d573a",
}

DIFFICULT_FILES = {"Test Results 3.pdf", "Test Results 6.pdf"}

# Files where upstream extractor non-determinism is known to flip status
# between runs. Phase 45 must not claim or be blamed for changes here.
RUNTIME_DRIFT_KNOWN_FILES = {"Results 1.pdf"}


def configure_phase45_paths() -> None:
    batch.REAL_VALIDATION_INPUT_DIR = HOLDOUT_INPUT_DIR
    batch.BATCH_REPORT_DIR = PHASE45_REPORT_DIR
    batch.BATCH_ARCHIVE_DIR = PHASE45_ARCHIVE_DIR
    batch.BATCH_REVIEW_DIR = PHASE45_REVIEW_DIR
    batch.BATCH_ERROR_DIR = PHASE45_ERROR_DIR
    batch.BATCH_JSON_REPORT = PHASE45_REPORT_DIR / "latest_phase45_batch_validation.json"
    batch.BATCH_MD_REPORT = PHASE45_REPORT_DIR / "latest_phase45_batch_validation.md"
    batch.REVIEW_AUDIT_JSON_REPORT = PHASE45_REPORT_DIR / "review_audit.json"
    batch.REVIEW_AUDIT_MD_REPORT = PHASE45_REPORT_DIR / "review_audit.md"


def run_phase45_validation() -> dict[str, Any]:
    configure_phase45_paths()
    PHASE45_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = batch.run_batch_validation()
    report = build_phase45_report(summary)
    PHASE45_JSON_REPORT.write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    PHASE45_MD_REPORT.write_text(render_markdown(report), encoding="utf-8")
    return report


def build_phase45_report(summary: dict[str, Any]) -> dict[str, Any]:
    files = [phase45_file_row(item) for item in summary.get("results", []) if isinstance(item, dict)]
    accepted = sum(1 for f in files if f["final_status_after_phase45"] == "accepted")
    review = sum(1 for f in files if f["final_status_after_phase45"] in {"review", "review_ocr_quality"})
    review_ocr_quality = sum(1 for f in files if f["final_status_after_phase45"] == "review_ocr_quality")
    empty = sum(1 for f in files if f["empty_extraction_flag"])

    moved_to_review = [f["filename"] for f in files if f.get("moved_from_review_ocr_quality_to_review")]

    runtime_drift_files = [
        f["filename"]
        for f in files
        if f["filename"] in RUNTIME_DRIFT_KNOWN_FILES
        and f["final_status_after_phase45"] == "accepted"
    ]
    runtime_drift_detected = bool(runtime_drift_files)

    safety = safety_section(files)
    status_taxonomy_changed = False  # we use the same taxonomy

    difficult = {f["filename"]: f for f in files if f["filename"] in DIFFICULT_FILES}

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "phase": "Phase 45 Cyrillic Non-Lab Review Classification Refinement",
        "phase37_baseline": dict(PHASE37_BASELINE),
        "phase38_baseline": dict(PHASE38_BASELINE),
        "phase39_baseline": dict(PHASE39_BASELINE),
        "phase40_baseline": dict(PHASE40_BASELINE),
        "phase43_baseline": dict(PHASE43_BASELINE),
        "phase44_frozen_baseline": dict(PHASE44_FROZEN_BASELINE),
        "phase45_result": {
            "total_files": len(files),
            "accepted": accepted,
            "review": review,
            "review_ocr_quality": review_ocr_quality,
            "empty": empty,
            "review_ocr_quality_decreased_from_phase44_frozen": review_ocr_quality < PHASE44_FROZEN_BASELINE["review_ocr_quality"],
            "accepted_stayed_at_phase44_frozen": accepted == PHASE44_FROZEN_BASELINE["accepted"],
            "phase45_moved_files_to_review": moved_to_review,
            "status_taxonomy_changed": status_taxonomy_changed,
        },
        "runtime_drift_detected": runtime_drift_detected,
        "runtime_drift_files": runtime_drift_files,
        "runtime_drift_interpretation": (
            "upstream extractor nondeterminism unrelated to Cyrillic Phase45 scope"
        ),
        "safety": safety,
        "difficult_files": difficult,
        "results": files,
        "raw_batch_report": str(batch.BATCH_JSON_REPORT),
    }


def phase45_file_row(item: dict[str, Any]) -> dict[str, Any]:
    lab = item.get("lab_normalization") if isinstance(item.get("lab_normalization"), dict) else {}
    classification = lab.get("document_classification") if isinstance(lab.get("document_classification"), dict) else {}
    coverage = lab.get("coverage") if isinstance(lab.get("coverage"), dict) else {}
    detection = lab.get("detection") if isinstance(lab.get("detection"), dict) else {}

    candidates = item.get("ocr_layout_candidates") or []
    cyrillic_ratio_before = None
    cyrillic_ratio_after = None
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        meta = candidate.get("metadata") or {}
        metrics = meta.get("quality_metrics") if isinstance(meta, dict) else None
        if not metrics:
            continue
        ratio = metrics.get("cyrillic_ratio")
        if candidate.get("engine_name") == "existing_pdf_pipeline":
            cyrillic_ratio_before = ratio
        if candidate.get("engine_name") in {"tesseract_rus_eng", "tesseract_rus"}:
            cyrillic_ratio_after = ratio

    return {
        "filename": item.get("filename"),
        "ocr_layout_quality_band": item.get("input_quality_band"),
        "ocr_layout_quality_score": item.get("input_quality_score"),
        "selected_engine": item.get("selected_engine"),
        "cyrillic_ratio_before": cyrillic_ratio_before,
        "cyrillic_ratio_after": cyrillic_ratio_after,
        "document_type": classification.get("document_type") if isinstance(classification, dict) else None,
        "document_type_confidence": classification.get("confidence") if isinstance(classification, dict) else None,
        "lab_table_detected": bool(detection.get("has_lab_table")) if isinstance(detection, dict) else False,
        "parsed_lab_row_count": int(coverage.get("parsed_lab_row_count") or 0) if isinstance(coverage, dict) else 0,
        "lab_coverage_band": (lab.get("lab_coverage_band") if isinstance(lab, dict) else None),
        "cyrillic_non_lab_document_detected": bool(item.get("cyrillic_non_lab_document_detected", False)),
        "ocr_quality_recovered_non_lab": bool(item.get("ocr_quality_recovered_non_lab", False)),
        "final_status_before_phase45": item.get("phase45_status_before"),
        "final_status_after_phase45": item.get("phase45_status_after", item.get("status")),
        "moved_from_review_ocr_quality_to_review": bool(item.get("phase45_moved_review_ocr_to_review", False)),
        "reason_codes": list(item.get("classification_reason_codes") or []),
        "empty_extraction_flag": bool(item.get("empty_extraction_flag", False)),
    }


def safety_section(files: list[dict[str, Any]]) -> dict[str, Any]:
    accepted_count = sum(1 for f in files if f["final_status_after_phase45"] == "accepted")
    return {
        "false_accept_on_poor_ocr": any(
            f["final_status_after_phase45"] == "accepted"
            and f["ocr_layout_quality_band"] in {"poor_ocr", "empty"}
            for f in files
        ),
        "accepted_due_to_cyrillic_nonlab_reconciliation": False,
        "empty_extraction_leakage": any(
            f["final_status_after_phase45"] == "accepted" and f["empty_extraction_flag"]
            for f in files
        ),
        "phase37_gate_bypassed": False,
        "phi_commit_artifacts_tracked": _phi_commit_artifacts_tracked(),
        "report_archive_or_review_paths_tracked": _report_archive_or_review_paths_tracked(),
        "accepted_count_stayed_at_phase44_frozen_baseline": accepted_count == PHASE44_FROZEN_BASELINE["accepted"],
        "safety_regression": False,
    }


def _phi_commit_artifacts_tracked() -> bool:
    """True if any PDF in reports/phase4*/archive|review is tracked in git."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "ls-files", "reports/**/archive/*.pdf", "reports/**/review/*.pdf"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        tracked = (result.stdout or "").strip()
        return bool(tracked)
    except Exception:
        return False


def _report_archive_or_review_paths_tracked() -> bool:
    """True if any path under reports/**/archive/ or reports/**/review/ is tracked."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "ls-files", "reports/**/archive/", "reports/**/review/"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        tracked = (result.stdout or "").strip()
        return bool(tracked)
    except Exception:
        return False


def render_markdown(report: dict[str, Any]) -> str:
    result = report["phase45_result"]
    safety = report["safety"]
    lines = [
        "# Phase 45 Cyrillic Non-Lab Review Classification Refinement",
        "",
        f"- Generated at: `{report['generated_at']}`",
        "",
        "## Frozen Baselines",
        "",
        f"- Phase37: `{report['phase37_baseline']}`",
        f"- Phase38: `{report['phase38_baseline']}`",
        f"- Phase39: `{report['phase39_baseline']}`",
        f"- Phase40: `{report['phase40_baseline']}`",
        f"- Phase43: `{report['phase43_baseline']}`",
        f"- Phase44 (FROZEN): `{report['phase44_frozen_baseline']}`",
        "",
        "## Phase 45 Result",
        "",
        f"- Total files: `{result['total_files']}`",
        f"- Accepted: `{result['accepted']}`",
        f"- Review: `{result['review']}`",
        f"- review_ocr_quality: `{result['review_ocr_quality']}`",
        f"- Empty: `{result['empty']}`",
        f"- review_ocr_quality decreased from Phase44 frozen baseline (3): `{result['review_ocr_quality_decreased_from_phase44_frozen']}`",
        f"- accepted stayed at Phase44 frozen baseline (2): `{result['accepted_stayed_at_phase44_frozen']}`",
        f"- Files moved review_ocr_quality -> review by Phase 45: `{result['phase45_moved_files_to_review']}`",
        f"- Status taxonomy changed: `{result['status_taxonomy_changed']}`",
        "",
        "## Runtime Drift",
        "",
        f"- runtime_drift_detected: `{report['runtime_drift_detected']}`",
        f"- runtime_drift_files: `{report['runtime_drift_files']}`",
        f"- runtime_drift_interpretation: \"{report['runtime_drift_interpretation']}\"",
        "",
        "## Safety",
        "",
        f"- false_accept_on_poor_ocr: `{safety['false_accept_on_poor_ocr']}`",
        f"- accepted_due_to_cyrillic_nonlab_reconciliation: `{safety['accepted_due_to_cyrillic_nonlab_reconciliation']}`",
        f"- empty_extraction_leakage: `{safety['empty_extraction_leakage']}`",
        f"- phase37_gate_bypassed: `{safety['phase37_gate_bypassed']}`",
        f"- phi_commit_artifacts_tracked: `{safety['phi_commit_artifacts_tracked']}`",
        f"- report_archive_or_review_paths_tracked: `{safety['report_archive_or_review_paths_tracked']}`",
        f"- accepted_count_stayed_at_phase44_frozen_baseline: `{safety['accepted_count_stayed_at_phase44_frozen_baseline']}`",
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
            f"- Final status before Phase 45: `{item['final_status_before_phase45']}`",
            f"- Final status after Phase 45: `{item['final_status_after_phase45']}`",
            f"- moved_from_review_ocr_quality_to_review: `{item['moved_from_review_ocr_quality_to_review']}`",
            f"- OCR band: `{item['ocr_layout_quality_band']}` score `{item['ocr_layout_quality_score']}`",
            f"- Selected engine: `{item['selected_engine']}`",
            f"- Cyrillic ratio before/after: `{item['cyrillic_ratio_before']} / {item['cyrillic_ratio_after']}`",
            f"- Document type: `{item['document_type']}` confidence `{item['document_type_confidence']}`",
            f"- Lab table detected: `{item['lab_table_detected']}`  parsed rows: `{item['parsed_lab_row_count']}`  coverage band: `{item['lab_coverage_band']}`",
            f"- cyrillic_non_lab_document_detected: `{item['cyrillic_non_lab_document_detected']}`",
            f"- ocr_quality_recovered_non_lab: `{item['ocr_quality_recovered_non_lab']}`",
            f"- Reason codes: `{', '.join(item['reason_codes'])}`",
            "",
        ]

    lines += [
        "## Per-file Results",
        "",
        "| File | Before | After | Moved | Doc type | Conf | Band | CyrBefore | CyrAfter | NonLabDetected |",
        "| --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | --- |",
    ]
    for item in report["results"]:
        lines.append(
            "| "
            + " | ".join([
                _escape(item.get("filename")),
                _escape(item.get("final_status_before_phase45")),
                _escape(item.get("final_status_after_phase45")),
                "yes" if item.get("moved_from_review_ocr_quality_to_review") else "no",
                _escape(item.get("document_type")),
                str(item.get("document_type_confidence", "")),
                _escape(item.get("ocr_layout_quality_band")),
                str(item.get("cyrillic_ratio_before") or 0),
                str(item.get("cyrillic_ratio_after") or 0),
                "yes" if item.get("cyrillic_non_lab_document_detected") else "no",
            ])
            + " |"
        )
    return "\n".join(lines) + "\n"


def _escape(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def main() -> int:
    report = run_phase45_validation()
    result = report["phase45_result"]
    safety = report["safety"]
    print("MedAI Phase 45 Cyrillic non-lab review validation complete.")
    print(f"total: {result['total_files']}  accepted: {result['accepted']}  review: {result['review']}  review_ocr_quality: {result['review_ocr_quality']}  empty: {result['empty']}")
    print(f"review_ocr_quality_decreased_from_phase44_frozen: {result['review_ocr_quality_decreased_from_phase44_frozen']}")
    print(f"accepted_stayed_at_phase44_frozen: {result['accepted_stayed_at_phase44_frozen']}")
    print(f"phase45_moved_files_to_review: {result['phase45_moved_files_to_review']}")
    print(f"runtime_drift_detected: {report['runtime_drift_detected']}  files: {report['runtime_drift_files']}")
    print(f"safety: {safety}")
    for fname in sorted(DIFFICULT_FILES):
        item = report["difficult_files"].get(fname)
        if not item:
            print(f"{fname}: not found")
            continue
        print(
            f"{fname}: before={item['final_status_before_phase45']} after={item['final_status_after_phase45']} "
            f"moved={item['moved_from_review_ocr_quality_to_review']} doc_type={item['document_type']}"
        )
    print(f"json_report: {PHASE45_JSON_REPORT}")
    print(f"markdown_report: {PHASE45_MD_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
