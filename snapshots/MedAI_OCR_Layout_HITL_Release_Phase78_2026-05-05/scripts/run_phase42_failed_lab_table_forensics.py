from __future__ import annotations

"""Phase 42 — Failed Lab Table Forensics.

For each persistent review_ocr_quality file, capture the OCR/Layout
selection, the production classification, and a forensic breakdown of
what the lab row parser saw and why each candidate line was rejected.
This is a diagnostic phase — it does not change parsing behavior.
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution.pipeline import ExecutionPipeline
from lab_normalization.lab_normalizer import normalize_lab_text
from lab_normalization.lab_row_parser import debug_parse_lab_lines
import scripts.run_batch_validation as batch


HOLDOUT_INPUT_DIR = ROOT / "holdout_validation_input"
PHASE42_REPORT_DIR = ROOT / "reports" / "phase42_failed_lab_table_forensics"
PHASE42_JSON_REPORT = PHASE42_REPORT_DIR / "phase42_failed_lab_table_forensics_report.json"
PHASE42_MD_REPORT = PHASE42_REPORT_DIR / "phase42_failed_lab_table_forensics_report.md"

TARGET_FILES = ["Test Results 3.pdf", "Test Results 6.pdf"]


def run_forensics() -> dict[str, Any]:
    PHASE42_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pipeline = ExecutionPipeline()
    files_report: list[dict[str, Any]] = []
    for fname in TARGET_FILES:
        source_path = HOLDOUT_INPUT_DIR / fname
        if not source_path.exists():
            files_report.append({
                "filename": fname,
                "error": "file not found in holdout_validation_input",
            })
            continue
        files_report.append(forensics_for_file(pipeline, source_path))
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "phase": "Phase 42 Failed Lab Table Forensics",
        "targets": TARGET_FILES,
        "files": files_report,
        "summary": global_summary(files_report),
    }
    PHASE42_JSON_REPORT.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    PHASE42_MD_REPORT.write_text(render_markdown(report), encoding="utf-8")
    return report


def _json_default(obj: Any) -> Any:
    try:
        return str(obj)
    except Exception:
        return repr(obj)


def forensics_for_file(pipeline: ExecutionPipeline, source_path: Path) -> dict[str, Any]:
    ocr = batch.build_ocr_layout_context(source_path)
    selected_text = str(ocr.get("selected_text") or "")
    classification = run_classification(pipeline, source_path, ocr, selected_text)
    debug = debug_parse_lab_lines(selected_text)
    lab_recovery = normalize_lab_text(
        selected_text,
        ocr_layout_quality_band=str(ocr.get("input_quality_band") or "unknown"),
        current_status=classification["status"],
        entity_count=classification["entity_count"],
        safety_gate_blocked=False,
    )
    diagnosis = diagnose_bottleneck(
        ocr=ocr,
        classification=classification,
        debug=debug,
        lab_recovery=lab_recovery.to_dict(),
    )
    return {
        "filename": source_path.name,
        "ocr_layout": {
            "selected_engine": ocr.get("selected_engine"),
            "input_quality_score": ocr.get("input_quality_score"),
            "input_quality_band": ocr.get("input_quality_band"),
            "route_decision": ocr.get("route_decision"),
            "input_quality_warnings": list(ocr.get("input_quality_warnings") or []),
        },
        "raw_text_length": len(selected_text.strip()),
        "raw_text_preview": selected_text.strip()[:500],
        "classification": classification,
        "debug_parse": debug,
        "lab_normalization": {
            "lab_table_detected": lab_recovery.lab_table_detected,
            "lab_coverage_ratio": lab_recovery.lab_coverage_ratio,
            "lab_coverage_band": lab_recovery.lab_coverage_band,
            "lab_reason_codes": lab_recovery.lab_reason_codes,
            "should_upgrade_from_ocr_review_to_review": lab_recovery.should_upgrade_from_ocr_review_to_review,
            "safety_checks": lab_recovery.safety_checks,
            "detection": lab_recovery.detection,
        },
        "diagnosis": diagnosis,
    }


def run_classification(
    pipeline: ExecutionPipeline,
    source_path: Path,
    ocr: dict[str, Any],
    selected_text: str,
) -> dict[str, Any]:
    """Mirror the production classification path used by run_batch_validation."""
    text_diagnostics = batch.analyze_text(selected_text, method=str(ocr.get("selected_engine") or "ocr_layout"))
    text_diagnostics = batch.diagnostics_from_ocr_layout(ocr, text_diagnostics)
    result = pipeline.process_text(
        selected_text,
        specialty="general",
        source_name=source_path.name,
        session_id=f"phase42_{source_path.stem}",
    )
    extractor_result = dict(result.extractor_result or {})
    audit = dict(result.audit or {})
    entities = list(extractor_result.get("entities", []))
    confidence = batch.safe_float(extractor_result.get("confidence", audit.get("confidence")))
    confidence_breakdown = extractor_result.get("confidence_breakdown")
    review_reason = batch.review_reason_for(result, extractor_result)
    normalization_applied = bool(extractor_result.get("normalization_applied", False))
    legacy_codes = batch.detect_ocr_low_quality_reason_codes(
        text_diagnostics=text_diagnostics,
        normalization_applied=normalization_applied,
        confidence_breakdown=confidence_breakdown,
    )
    is_ocr_low_quality = bool(legacy_codes) or batch.ocr_layout_forces_ocr_review(ocr)
    status = batch.classify_batch_status(
        outcome=result.outcome,
        review_reason=review_reason,
        confidence=confidence,
        entity_count=len(entities),
        is_ocr_low_quality=is_ocr_low_quality,
    )
    why_reviewed = batch.review_reasons_for(
        status=status,
        entity_count=len(entities),
        confidence=confidence,
        confidence_breakdown=confidence_breakdown,
    )
    classification_reasons = batch.classification_reason_codes_for(
        status=status,
        entity_count=len(entities),
        confidence=confidence,
        confidence_breakdown=confidence_breakdown,
        review_reason=review_reason,
        why_reviewed=why_reviewed,
        ocr_layout=ocr,
        legacy_ocr_reason_codes=legacy_codes,
    )
    return {
        "status": status,
        "entity_count": len(entities),
        "confidence": confidence,
        "confidence_breakdown": confidence_breakdown,
        "why_reviewed": why_reviewed,
        "review_reason": review_reason,
        "is_ocr_low_quality": is_ocr_low_quality,
        "legacy_ocr_reason_codes": legacy_codes,
        "classification_reason_codes": classification_reasons,
    }


def diagnose_bottleneck(
    *,
    ocr: dict[str, Any],
    classification: dict[str, Any],
    debug: dict[str, Any],
    lab_recovery: dict[str, Any],
) -> dict[str, Any]:
    """Pick a primary bottleneck category (A–E) for this file.

    A. OCR/Layout candidate generation
    B. Table block segmentation
    C. Row parser
    D. Classifier threshold/reasoning
    E. True manual-review boundary
    """
    band = str(ocr.get("input_quality_band") or "unknown")
    candidate_count = int(debug.get("candidate_line_count") or 0)
    parsed_count = int(debug.get("parsed_row_count") or 0)
    rejected_count = int(debug.get("rejected_line_count") or 0)
    breakdown = dict(debug.get("rejection_reason_breakdown") or {})
    signals = dict(debug.get("signal_summary") or {})
    coverage = float(lab_recovery.get("lab_coverage_ratio") or 0.0)
    table_detected = bool(lab_recovery.get("lab_table_detected"))
    legacy = list(classification.get("legacy_ocr_reason_codes") or [])

    notes: list[str] = []
    category = "E"
    rationale = ""

    if band in {"poor_ocr", "empty"} or candidate_count == 0:
        category = "A"
        rationale = "OCR/Layout produced empty or unusable text — candidate generation is the bottleneck."
    elif signals.get("text_too_sparse") or signals.get("text_too_fragmented"):
        category = "A"
        rationale = "Selected text is sparse or fragmented — OCR/Layout is dropping lines before parsing."
    elif not table_detected and candidate_count >= 5:
        category = "B"
        rationale = "Lines exist but lab_table_detector did not recognize a table — segmentation is the bottleneck."
    elif (
        signals.get("has_name_only_lines")
        and (signals.get("has_value_only_lines") or signals.get("has_range_only_lines"))
        and parsed_count <= 1
    ):
        category = "B"
        rationale = (
            "Names, values, and ranges appear on separate lines — table block segmentation "
            "is splitting one row across many lines."
        )
    elif rejected_count >= 3 and breakdown.get("malformed_lab_row", 0) + breakdown.get("no_unit_or_qualitative", 0) >= 3:
        category = "C"
        rationale = "Multiple lines look like rows but failed parser regexes — row parser is the bottleneck."
    elif (
        parsed_count >= 1
        and coverage < 0.10
        and band in {"good", "usable_with_review"}
        and "legacy_normalized_low_coverage" in legacy
    ):
        category = "D"
        rationale = (
            "Some rows parse but classifier still flags review_ocr_quality due to "
            "legacy_normalized_low_coverage — classifier reasoning is the bottleneck."
        )
    elif parsed_count == 0 and band in {"good", "usable_with_review"} and not signals.get("appears_table_like_no_separators"):
        category = "E"
        rationale = "OCR gave clean text but no structured lab table content was present — likely a true manual-review boundary."
    else:
        category = "C"
        rationale = "Default: candidate lines exist but the parser produced too few rows."

    if signals.get("appears_table_like_no_separators"):
        notes.append("appears_table_like_no_separators=True — multi-column layout without | or \\t separators.")
    if signals.get("has_range_only_lines"):
        notes.append("range-only lines present (Phase 41 split-range covers consecutive pairs only).")
    if breakdown:
        notes.append(f"rejection breakdown: {breakdown}")

    return {
        "primary_bottleneck_category": category,
        "category_legend": {
            "A": "OCR/Layout candidate generation",
            "B": "Table block segmentation",
            "C": "Row parser",
            "D": "Classifier threshold/reasoning",
            "E": "True manual-review boundary",
        },
        "rationale": rationale,
        "notes": notes,
    }


def global_summary(files: list[dict[str, Any]]) -> dict[str, Any]:
    categories = [
        f.get("diagnosis", {}).get("primary_bottleneck_category")
        for f in files
        if "diagnosis" in f
    ]
    distribution: dict[str, int] = {}
    for c in categories:
        if c is None:
            continue
        distribution[c] = distribution.get(c, 0) + 1
    return {
        "files_analyzed": sum(1 for f in files if "diagnosis" in f),
        "files_missing": sum(1 for f in files if "error" in f),
        "category_distribution": distribution,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 42 Failed Lab Table Forensics",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Targets: {', '.join(f'`{t}`' for t in report['targets'])}",
        "",
        "## Summary",
        "",
        f"- Files analyzed: `{report['summary']['files_analyzed']}`",
        f"- Files missing: `{report['summary']['files_missing']}`",
        f"- Bottleneck category distribution: `{report['summary']['category_distribution']}`",
        "",
        "## Category Legend",
        "",
        "- **A** — OCR/Layout candidate generation",
        "- **B** — Table block segmentation",
        "- **C** — Row parser",
        "- **D** — Classifier threshold/reasoning",
        "- **E** — True manual-review boundary",
        "",
    ]
    for entry in report["files"]:
        if "error" in entry:
            lines += [
                f"## {entry['filename']}",
                "",
                f"- ERROR: {entry['error']}",
                "",
            ]
            continue
        ocr = entry["ocr_layout"]
        cls = entry["classification"]
        debug = entry["debug_parse"]
        lab = entry["lab_normalization"]
        diag = entry["diagnosis"]
        signals = debug.get("signal_summary", {})
        breakdown = debug.get("rejection_reason_breakdown", {})

        lines += [
            f"## {entry['filename']}",
            "",
            "### OCR/Layout",
            "",
            f"- Engine: `{ocr.get('selected_engine')}`",
            f"- Quality score: `{ocr.get('input_quality_score')}`",
            f"- Quality band: `{ocr.get('input_quality_band')}`",
            f"- Route decision: `{ocr.get('route_decision')}`",
            f"- Quality warnings: `{ocr.get('input_quality_warnings')}`",
            f"- Raw text length: `{entry.get('raw_text_length')}`",
            "",
            "### Classification",
            "",
            f"- Final status: `{cls.get('status')}`",
            f"- Entity count: `{cls.get('entity_count')}`",
            f"- Confidence: `{cls.get('confidence')}`",
            f"- Reason codes: `{', '.join(cls.get('classification_reason_codes') or [])}`",
            f"- Legacy OCR codes: `{', '.join(cls.get('legacy_ocr_reason_codes') or [])}`",
            "",
            "### Lab Normalization",
            "",
            f"- Lab table detected: `{lab.get('lab_table_detected')}`",
            f"- Coverage ratio: `{lab.get('lab_coverage_ratio')}`",
            f"- Coverage band: `{lab.get('lab_coverage_band')}`",
            f"- Reason codes: `{', '.join(lab.get('lab_reason_codes') or [])}`",
            f"- Would upgrade review_ocr_quality→review: `{lab.get('should_upgrade_from_ocr_review_to_review')}`",
            "",
            "### Forensic Parse",
            "",
            f"- Candidate lines: `{debug.get('candidate_line_count')}`",
            f"- Parsed rows: `{debug.get('parsed_row_count')}`",
            f"- Rejected lines: `{debug.get('rejected_line_count')}`",
            f"- Rejection breakdown: `{breakdown}`",
            "",
            "#### Signals",
            "",
            f"- value-only lines present: `{signals.get('has_value_only_lines')}`",
            f"- name-only lines present: `{signals.get('has_name_only_lines')}`",
            f"- range-only lines present: `{signals.get('has_range_only_lines')}`",
            f"- appears table-like without separators: `{signals.get('appears_table_like_no_separators')}`",
            f"- text too fragmented (>50% short lines): `{signals.get('text_too_fragmented')}`",
            f"- text too sparse (<3 lines): `{signals.get('text_too_sparse')}`",
            f"- short_line_ratio: `{signals.get('short_line_ratio')}`",
            "",
            "#### Top candidate lines (up to 50)",
            "",
            "```",
            *(_truncate(line) for line in (debug.get("candidate_lines") or [])),
            "```",
            "",
            "#### Rejected lines",
            "",
            "| # | reason | partial | line |",
            "| ---: | --- | --- | --- |",
        ]
        for r in debug.get("rejected_lines") or []:
            lines.append(
                "| "
                + " | ".join([
                    str(r.get("index")),
                    _escape_md(r.get("reason")),
                    _escape_md(r.get("matched_partial_pattern")),
                    _escape_md(_truncate(r.get("line") or "")),
                ])
                + " |"
            )

        lines += [
            "",
            "### Diagnosis",
            "",
            f"- **Primary bottleneck:** `{diag.get('primary_bottleneck_category')}` "
            f"({diag.get('category_legend', {}).get(diag.get('primary_bottleneck_category'), '')})",
            f"- Rationale: {diag.get('rationale')}",
            "",
        ]
        for note in diag.get("notes") or []:
            lines.append(f"- Note: {note}")
        lines.append("")

        lines += [
            "### Raw text preview",
            "",
            "```",
            _truncate(entry.get("raw_text_preview") or "", limit=500),
            "```",
            "",
        ]

    return "\n".join(lines) + "\n"


def _truncate(value: str, *, limit: int = 200) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _escape_md(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def main() -> int:
    report = run_forensics()
    print("MedAI Phase 42 failed lab table forensics complete.")
    print(f"files_analyzed: {report['summary']['files_analyzed']}")
    print(f"files_missing: {report['summary']['files_missing']}")
    print(f"category_distribution: {report['summary']['category_distribution']}")
    for entry in report["files"]:
        if "error" in entry:
            print(f"{entry['filename']}: ERROR — {entry['error']}")
            continue
        diag = entry["diagnosis"]
        debug = entry["debug_parse"]
        cls = entry["classification"]
        print(
            f"{entry['filename']}: status={cls.get('status')} "
            f"candidates={debug.get('candidate_line_count')} parsed={debug.get('parsed_row_count')} "
            f"category={diag.get('primary_bottleneck_category')}"
        )
    print(f"json_report: {PHASE42_JSON_REPORT}")
    print(f"markdown_report: {PHASE42_MD_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
