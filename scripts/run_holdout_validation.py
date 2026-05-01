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
HOLDOUT_REPORT_DIR = ROOT / "reports" / "holdout_validation"
HOLDOUT_ARCHIVE_DIR = HOLDOUT_REPORT_DIR / "archive"
HOLDOUT_REVIEW_DIR = HOLDOUT_REPORT_DIR / "review"
HOLDOUT_ERROR_DIR = HOLDOUT_REPORT_DIR / "error"
HOLDOUT_JSON_REPORT = HOLDOUT_REPORT_DIR / "latest_holdout_validation.json"
HOLDOUT_MD_REPORT = HOLDOUT_REPORT_DIR / "latest_holdout_validation.md"
HOLDOUT_REVIEW_AUDIT_JSON = HOLDOUT_REPORT_DIR / "review_audit.json"
HOLDOUT_REVIEW_AUDIT_MD = HOLDOUT_REPORT_DIR / "review_audit.md"
HOLDOUT_PHASE35_AUDIT_JSON = HOLDOUT_REPORT_DIR / "review_audit_phase35.json"
HOLDOUT_PHASE35_AUDIT_MD = HOLDOUT_REPORT_DIR / "review_audit_phase35.md"
HOLDOUT_COMPARISON_REPORT = HOLDOUT_REPORT_DIR / "comparison_to_baseline.md"
BASELINE_JSON_REPORT = ROOT / "reports" / "baseline_phase33" / "baseline_metrics.json"
PHASE35_ISSUE_CATEGORIES = (
    "ocr_noise_remaining",
    "low_coverage_after_normalization",
    "missing_domain_rules",
    "correctly_flagged_review",
)


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
    write_phase35_review_audit()
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


def write_phase35_review_audit() -> tuple[Path, Path]:
    if HOLDOUT_JSON_REPORT.exists():
        summary = json.loads(HOLDOUT_JSON_REPORT.read_text(encoding="utf-8"))
    else:
        summary = {"timestamp": None, "results": []}
    audit = build_phase35_review_audit(summary)
    HOLDOUT_PHASE35_AUDIT_JSON.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    HOLDOUT_PHASE35_AUDIT_MD.write_text(render_phase35_review_audit_markdown(audit), encoding="utf-8")
    return HOLDOUT_PHASE35_AUDIT_JSON, HOLDOUT_PHASE35_AUDIT_MD


def build_phase35_review_audit(summary: dict[str, Any]) -> dict[str, Any]:
    reviewed = [item for item in summary.get("results", []) if item.get("status") == "review"]
    files = [phase35_review_item(item) for item in reviewed]
    breakdown = {category: 0 for category in PHASE35_ISSUE_CATEGORIES}
    for item in files:
        category = item["remaining_issue_category"]
        if category in breakdown:
            breakdown[category] += 1
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_report": str(HOLDOUT_JSON_REPORT),
        "source_timestamp": summary.get("timestamp"),
        "total_reviewed": len(files),
        "remaining_issue_breakdown": breakdown,
        "files": files,
    }


def phase35_review_item(item: dict[str, Any]) -> dict[str, Any]:
    diagnostics = item.get("text_diagnostics") if isinstance(item.get("text_diagnostics"), dict) else {}
    confidence_breakdown = item.get("confidence_breakdown") if isinstance(item.get("confidence_breakdown"), dict) else {}
    normalized_breakdown = batch.normalized_review_confidence_breakdown(confidence_breakdown)
    text_preview = str(diagnostics.get("preview") or "")[:300]
    normalized_preview = str(item.get("normalized_text_preview") or "")[:300]
    normalization_applied = bool(item.get("normalization_applied", False))
    return {
        "filename": item.get("filename"),
        "entity_count": int(item.get("entity_count") or 0),
        "entities": list(item.get("entities") or []),
        "confidence": batch.safe_float(item.get("confidence")),
        "confidence_breakdown": normalized_breakdown,
        "why_reviewed": list(item.get("why_reviewed") or []),
        "normalization_applied": normalization_applied,
        "normalized_text_preview": normalized_preview,
        "text_preview": text_preview,
        "text_length": int(diagnostics.get("length") or 0),
        "extraction_method": diagnostics.get("method"),
        "remaining_issue_category": classify_phase35_remaining_issue(
            item=item,
            diagnostics=diagnostics,
            coverage_score=normalized_breakdown["coverage_score"],
            text_preview=text_preview,
            normalization_applied=normalization_applied,
        ),
    }


def classify_phase35_remaining_issue(
    *,
    item: dict[str, Any],
    diagnostics: dict[str, Any],
    coverage_score: float | None,
    text_preview: str,
    normalization_applied: bool,
) -> str:
    suspicious = bool(diagnostics.get("suspicious", False))
    low_coverage = coverage_score is not None and coverage_score < 0.3
    if suspicious or (normalization_applied and low_coverage):
        return "ocr_noise_remaining"
    if low_coverage:
        return "low_coverage_after_normalization"
    if int(item.get("entity_count") or 0) <= 1 and looks_like_medical_text(text_preview):
        return "missing_domain_rules"
    return "correctly_flagged_review"


def looks_like_medical_text(text: str) -> bool:
    normalized = text.lower()
    indicators = (
        "urine",
        "blood",
        "culture",
        "diagnosis",
        "cytology",
        "glucose",
        "protein",
        "nitrite",
        "rbc",
        "wbc",
        "patient",
        "specimen",
        "report",
    )
    return any(indicator in normalized for indicator in indicators)


def render_phase35_review_audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Phase 35 Holdout Review Audit",
        "",
        f"- Source report: `{audit.get('source_report')}`",
        f"- Reviewed files: `{audit.get('total_reviewed', 0)}`",
        "",
        "## Remaining Issue Breakdown",
        "",
    ]
    breakdown = audit.get("remaining_issue_breakdown") or {}
    lines.extend(f"- {category}: `{breakdown.get(category, 0)}`" for category in PHASE35_ISSUE_CATEGORIES)

    for item in audit.get("files", []):
        entity_texts = [
            str(entity.get("text", ""))
            for entity in item.get("entities", [])
            if isinstance(entity, dict) and str(entity.get("text", "")).strip()
        ]
        lines.extend([
            "",
            f"## {item.get('filename')}",
            "",
            f"- remaining issue: {item.get('remaining_issue_category')}",
            f"- confidence: {item.get('confidence')}",
            f"- entity count: {item.get('entity_count')}",
            f"- entities: {entity_texts}",
            f"- why reviewed: {item.get('why_reviewed') or []}",
            f"- normalization applied: {item.get('normalization_applied')}",
            f"- extraction method: {item.get('extraction_method')}",
            f"- text length: {item.get('text_length')}",
            "",
            "- normalized preview:",
            str(item.get("normalized_text_preview") or "").rstrip(),
            "",
            "- original preview:",
            str(item.get("text_preview") or "").rstrip(),
        ])
    return "\n".join(lines) + "\n"


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
    print(f"phase35_review_audit_json: {HOLDOUT_PHASE35_AUDIT_JSON}")
    print(f"phase35_review_audit_md: {HOLDOUT_PHASE35_AUDIT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
