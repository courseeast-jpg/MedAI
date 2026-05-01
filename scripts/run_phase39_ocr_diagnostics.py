from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PHASE38_BATCH_REPORT = ROOT / "reports" / "phase38_ocr_layout" / "latest_phase38_batch_validation.json"
PHASE38_SUMMARY_REPORT = ROOT / "reports" / "phase38_ocr_layout" / "phase38_ocr_layout_validation.json"
PHASE39_REPORT_DIR = ROOT / "reports" / "phase39_ocr_diagnostics"
PHASE39_JSON_REPORT = PHASE39_REPORT_DIR / "phase39_ocr_diagnostics_report.json"
PHASE39_MD_REPORT = PHASE39_REPORT_DIR / "phase39_ocr_diagnostics_report.md"

PHASE37_BASELINE = {
    "total_files": 8,
    "accepted": 2,
    "review_ocr_quality": 6,
    "empty": 0,
}

PHASE38_BASELINE = {
    "total_files": 8,
    "accepted": 2,
    "review": 6,
    "review_ocr_quality": 4,
    "empty": 0,
    "safety_regression": False,
}


def run_phase39_diagnostics() -> dict[str, Any]:
    PHASE39_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    phase38_batch = _load_json(PHASE38_BATCH_REPORT)
    phase38_summary = _load_json(PHASE38_SUMMARY_REPORT) if PHASE38_SUMMARY_REPORT.exists() else {}
    report = build_phase39_report(phase38_batch=phase38_batch, phase38_summary=phase38_summary)
    PHASE39_JSON_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    PHASE39_MD_REPORT.write_text(render_markdown(report), encoding="utf-8")
    return report


def build_phase39_report(*, phase38_batch: dict[str, Any], phase38_summary: dict[str, Any]) -> dict[str, Any]:
    results = [diagnostic_row(item) for item in phase38_batch.get("results", []) if isinstance(item, dict)]
    mismatches = [item for item in results if item["ocr_status_mismatch"]]
    review_ocr_quality = [item for item in results if item["downstream_classifier_status"] == "review_ocr_quality"]
    safety_regression = any(
        item["downstream_classifier_status"] == "accepted"
        and item["ocr_layout_quality_band"] in {"poor_ocr", "empty"}
        for item in results
    )
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "phase": "Phase 39 OCR/Layout Diagnostic Reconciliation",
        "source_phase38_batch_report": str(PHASE38_BATCH_REPORT),
        "phase37_baseline": dict(PHASE37_BASELINE),
        "phase38_baseline": {
            **PHASE38_BASELINE,
            "phase37_review_ocr_quality_improved_count": phase38_summary.get(
                "phase37_review_ocr_quality_improved_count",
                2,
            ),
        },
        "phase39_diagnostics": {
            "total_files": len(results),
            "ocr_status_mismatch_count": len(mismatches),
            "review_ocr_quality_count": len(review_ocr_quality),
            "safety_regression": safety_regression,
            "status_taxonomy_changed": False,
        },
        "mismatch_files": [item["filename"] for item in mismatches],
        "review_ocr_quality_reasons": {
            item["filename"]: item["downstream_classifier_reason"]
            for item in review_ocr_quality
        },
        "results": results,
        "phase40_recommendation": phase40_recommendation(results),
    }


def diagnostic_row(item: dict[str, Any]) -> dict[str, Any]:
    diagnostics = item.get("text_diagnostics") if isinstance(item.get("text_diagnostics"), dict) else {}
    return {
        "filename": item.get("filename"),
        "ocr_layout_quality_band": item.get("input_quality_band"),
        "ocr_layout_quality_score": item.get("input_quality_score"),
        "ocr_layout_route": item.get("route_decision"),
        "downstream_classifier_status": item.get("downstream_classifier_status", item.get("status")),
        "downstream_classifier_reason": item.get("downstream_classifier_reason"),
        "classification_reason_codes": list(item.get("classification_reason_codes") or []),
        "extraction_entity_count": int(item.get("entity_count") or 0),
        "confidence_score": item.get("confidence"),
        "empty_extraction_flag": bool(item.get("empty_extraction_flag", int(item.get("entity_count") or 0) == 0)),
        "table_layout_warning": bool(item.get("table_layout_warning", False)),
        "normalization_applied": bool(item.get("normalization_applied", False)),
        "ocr_status_mismatch": bool(item.get("ocr_status_mismatch", False)),
        "mismatch_type": item.get("mismatch_type"),
        "selected_extraction_engine": item.get("selected_engine"),
        "selected_extractor": item.get("selected_extractor"),
        "review_type": item.get("review_type"),
        "why_reviewed": list(item.get("why_reviewed") or []),
        "text_method": diagnostics.get("method"),
        "text_suspicious": bool(diagnostics.get("suspicious", False)),
    }


def phase40_recommendation(results: list[dict[str, Any]]) -> str:
    mismatches = [item for item in results if item["ocr_status_mismatch"]]
    if not mismatches:
        return "Continue improving OCR candidate quality; no status taxonomy change is indicated by Phase39 diagnostics."
    return (
        "Keep review_ocr_quality as the safe final status for now, but in Phase40 split the legacy normalized-low-coverage "
        "case into an extraction/table-structure diagnostic reason or review_extraction_quality path after downstream "
        "coverage metrics are calibrated against lab/table ground truth."
    )


def render_markdown(report: dict[str, Any]) -> str:
    phase37 = report["phase37_baseline"]
    phase38 = report["phase38_baseline"]
    diagnostics = report["phase39_diagnostics"]
    lines = [
        "# Phase 39 OCR/Layout Diagnostic Reconciliation",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Source Phase38 batch report: `{report['source_phase38_batch_report']}`",
        "",
        "## Baselines",
        "",
        f"- Phase37: total `{phase37['total_files']}`, accepted `{phase37['accepted']}`, review_ocr_quality `{phase37['review_ocr_quality']}`, empty `{phase37['empty']}`",
        f"- Phase38: total `{phase38['total_files']}`, accepted `{phase38['accepted']}`, review `{phase38['review']}`, review_ocr_quality `{phase38['review_ocr_quality']}`, empty `{phase38['empty']}`",
        f"- Phase38 improved prior review_ocr_quality files: `{phase38['phase37_review_ocr_quality_improved_count']}`",
        "",
        "## Phase39 Diagnostics",
        "",
        f"- Total files: `{diagnostics['total_files']}`",
        f"- OCR status mismatches: `{diagnostics['ocr_status_mismatch_count']}`",
        f"- review_ocr_quality files: `{diagnostics['review_ocr_quality_count']}`",
        f"- Safety regression: `{diagnostics['safety_regression']}`",
        f"- Status taxonomy changed: `{diagnostics['status_taxonomy_changed']}`",
        "",
        "## Mismatch Files",
        "",
    ]
    if report["mismatch_files"]:
        lines.extend(f"- `{filename}`" for filename in report["mismatch_files"])
    else:
        lines.append("- None")

    lines.extend([
        "",
        "## Per-file Diagnostic Table",
        "",
        "| File | OCR band | OCR score | OCR route | Final status | Final reason | Entities | Confidence | Empty | Table/layout warning | Mismatch |",
        "| --- | --- | ---: | --- | --- | --- | ---: | ---: | --- | --- | --- |",
    ])
    for item in report["results"]:
        lines.append(
            "| "
            + " | ".join([
                _escape_md(item.get("filename")),
                _escape_md(item.get("ocr_layout_quality_band")),
                "" if item.get("ocr_layout_quality_score") is None else str(item.get("ocr_layout_quality_score")),
                _escape_md(item.get("ocr_layout_route")),
                _escape_md(item.get("downstream_classifier_status")),
                _escape_md(item.get("downstream_classifier_reason")),
                str(item.get("extraction_entity_count", 0)),
                "" if item.get("confidence_score") is None else str(item.get("confidence_score")),
                "yes" if item.get("empty_extraction_flag") else "no",
                "yes" if item.get("table_layout_warning") else "no",
                _escape_md(item.get("mismatch_type")) if item.get("ocr_status_mismatch") else "no",
            ])
            + " |"
        )

    lines.extend([
        "",
        "## review_ocr_quality Reasons",
        "",
    ])
    for filename, reason in report["review_ocr_quality_reasons"].items():
        lines.append(f"- `{filename}`: `{reason}`")

    lines.extend([
        "",
        "## Phase40 Recommendation",
        "",
        report["phase40_recommendation"],
        "",
    ])
    return "\n".join(lines)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _escape_md(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def main() -> int:
    report = run_phase39_diagnostics()
    diagnostics = report["phase39_diagnostics"]
    print("MedAI Phase 39 OCR diagnostics complete.")
    print(f"total: {diagnostics['total_files']}")
    print(f"ocr_status_mismatches: {diagnostics['ocr_status_mismatch_count']}")
    print(f"review_ocr_quality: {diagnostics['review_ocr_quality_count']}")
    print(f"safety_regression: {diagnostics['safety_regression']}")
    print(f"status_taxonomy_changed: {diagnostics['status_taxonomy_changed']}")
    print(f"json_report: {PHASE39_JSON_REPORT}")
    print(f"markdown_report: {PHASE39_MD_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
