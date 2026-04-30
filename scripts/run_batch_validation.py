from __future__ import annotations

import json
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution.pipeline import ExecutionPipeline


REAL_VALIDATION_INPUT_DIR = ROOT / "real_validation_input"
BATCH_REPORT_DIR = ROOT / "reports" / "batch_validation"
BATCH_ARCHIVE_DIR = BATCH_REPORT_DIR / "archive"
BATCH_REVIEW_DIR = BATCH_REPORT_DIR / "review"
BATCH_ERROR_DIR = BATCH_REPORT_DIR / "error"
BATCH_JSON_REPORT = BATCH_REPORT_DIR / "latest_batch_validation.json"
BATCH_MD_REPORT = BATCH_REPORT_DIR / "latest_batch_validation.md"
REVIEW_AUDIT_JSON_REPORT = BATCH_REPORT_DIR / "review_audit.json"
REVIEW_AUDIT_MD_REPORT = BATCH_REPORT_DIR / "review_audit.md"
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}
ACCEPTED_OUTCOMES = {"written"}
REVIEW_REASON_KEYS = [
    "empty_extraction",
    "confidence_below_threshold",
    "low_entity_count",
    "low_coverage",
    "low_diversity",
    "low_extractor_weight",
]


REQUIRED_RESULT_FIELDS = [
    "filename",
    "status",
    "entity_count",
    "entities",
    "selected_extractor",
    "primary_extractor",
    "fallback_extractor",
    "fallback_reason",
    "terminal_empty_prevented",
    "confidence",
    "confidence_breakdown",
    "supplemental_rules_applied",
    "supplemental_entity_count",
    "final_entity_count_after_supplement",
    "review_reason",
    "why_reviewed",
    "error",
    "text_diagnostics",
]

OCR_ARTIFACT_PATTERN = re.compile(r"(\S)\1{7,}|[|_]{4,}|(?:\b[Il1]{8,}\b)|\ufffd")


def ensure_batch_validation_dirs() -> None:
    for path in (
        REAL_VALIDATION_INPUT_DIR,
        BATCH_REPORT_DIR,
        BATCH_ARCHIVE_DIR,
        BATCH_REVIEW_DIR,
        BATCH_ERROR_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def list_supported_input_files() -> list[Path]:
    ensure_batch_validation_dirs()
    return sorted(
        path
        for path in REAL_VALIDATION_INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def run_batch_validation(pipeline: ExecutionPipeline | None = None) -> dict[str, Any]:
    ensure_batch_validation_dirs()
    input_files = list_supported_input_files()
    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "run_id": str(uuid4()),
        "input_dir": str(REAL_VALIDATION_INPUT_DIR),
        "total_files": len(input_files),
        "accepted_count": 0,
        "review_count": 0,
        "error_count": 0,
        "empty_extraction_count": 0,
        "fallback_count": 0,
        "low_text_quality_count": 0,
        "avg_text_length": 0,
        "review_reason_summary": {reason: 0 for reason in REVIEW_REASON_KEYS},
        "results": [],
        "files_needing_review": [],
        "errors": [],
    }

    if not input_files:
        write_batch_reports(summary)
        return summary

    active_pipeline = pipeline or ExecutionPipeline()
    for source_path in input_files:
        result = process_one_file(active_pipeline, source_path, run_id=summary["run_id"])
        summary["results"].append(result)
        if result["status"] == "accepted":
            summary["accepted_count"] += 1
        elif result["status"] == "review":
            summary["review_count"] += 1
            summary["files_needing_review"].append(result["filename"])
            for reason in result.get("why_reviewed") or []:
                if reason in summary["review_reason_summary"]:
                    summary["review_reason_summary"][reason] += 1
        else:
            summary["error_count"] += 1
            summary["errors"].append({"filename": result["filename"], "error": result["error"]})
        if int(result["entity_count"]) == 0:
            summary["empty_extraction_count"] += 1
        if result["fallback_extractor"] or result["fallback_reason"] or result["terminal_empty_prevented"]:
            summary["fallback_count"] += 1
        diagnostics = result.get("text_diagnostics") or {}
        if diagnostics.get("suspicious"):
            summary["low_text_quality_count"] += 1

    text_lengths = [
        int((item.get("text_diagnostics") or {}).get("length", 0))
        for item in summary["results"]
    ]
    summary["avg_text_length"] = round(sum(text_lengths) / len(text_lengths), 2) if text_lengths else 0

    write_batch_reports(summary)
    return summary


def process_one_file(pipeline: ExecutionPipeline, source_path: Path, *, run_id: str) -> dict[str, Any]:
    text_diagnostics = build_initial_text_diagnostics(source_path)
    try:
        if source_path.suffix.lower() == ".pdf":
            result = pipeline.process_pdf(source_path, specialty="general", session_id=run_id)
        elif source_path.suffix.lower() == ".txt":
            source_text = source_path.read_text(encoding="utf-8", errors="replace")
            text_diagnostics = analyze_text(source_text, method="plain_text")
            result = pipeline.process_text(
                source_text,
                specialty="general",
                source_name=source_path.name,
                session_id=run_id,
            )
        else:
            raise ValueError(f"Unsupported file type: {source_path.suffix}")

        extractor_result = dict(result.extractor_result or {})
        audit = dict(result.audit or {})
        entities = list(extractor_result.get("entities", []))
        text_diagnostics = diagnostics_from_result(source_path, extractor_result, text_diagnostics)
        selected_extractor = (
            extractor_result.get("selected_extractor")
            or extractor_result.get("actual_extractor")
            or audit.get("extractor_actual")
            or audit.get("extractor")
        )
        review_reason = review_reason_for(result, extractor_result)
        confidence = safe_float(extractor_result.get("confidence", audit.get("confidence")))
        confidence_breakdown = extractor_result.get("confidence_breakdown")
        status = classify_batch_status(
            outcome=result.outcome,
            review_reason=review_reason,
            confidence=confidence,
            entity_count=len(entities),
        )
        copy_destination = copy_to_unique_destination(
            source_path,
            BATCH_ARCHIVE_DIR if status == "accepted" else BATCH_REVIEW_DIR,
        )
        why_reviewed = review_reasons_for(
            status=status,
            entity_count=len(entities),
            confidence=confidence,
            confidence_breakdown=confidence_breakdown,
        )
        return normalize_result({
            "filename": source_path.name,
            "status": status,
            "entity_count": len(entities),
            "entities": entities,
            "selected_extractor": str(selected_extractor) if selected_extractor is not None else None,
            "primary_extractor": extractor_result.get("primary_extractor"),
            "fallback_extractor": extractor_result.get("fallback_extractor"),
            "fallback_reason": extractor_result.get("fallback_reason"),
            "terminal_empty_prevented": bool(extractor_result.get("terminal_empty_prevented", False)),
            "confidence": confidence,
            "confidence_breakdown": confidence_breakdown,
            "supplemental_rules_applied": bool(extractor_result.get("supplemental_rules_applied", False)),
            "supplemental_entity_count": int(extractor_result.get("supplemental_entity_count", 0) or 0),
            "final_entity_count_after_supplement": int(
                extractor_result.get("final_entity_count_after_supplement", len(entities)) or 0
            ),
            "review_reason": review_reason,
            "why_reviewed": why_reviewed,
            "error": None,
            "text_diagnostics": text_diagnostics,
            "copied_to": str(copy_destination),
            "outcome": result.outcome,
            "validation_status": result.validation_status,
        })
    except Exception as exc:
        copy_destination = copy_to_unique_destination(source_path, BATCH_ERROR_DIR)
        return normalize_result({
            "filename": source_path.name,
            "status": "error",
            "entity_count": 0,
            "entities": [],
            "selected_extractor": None,
            "primary_extractor": None,
            "fallback_extractor": None,
            "fallback_reason": None,
            "terminal_empty_prevented": False,
            "confidence": None,
            "confidence_breakdown": None,
            "supplemental_rules_applied": False,
            "supplemental_entity_count": 0,
            "final_entity_count_after_supplement": 0,
            "review_reason": None,
            "why_reviewed": [],
            "error": str(exc),
            "text_diagnostics": text_diagnostics,
            "copied_to": str(copy_destination),
            "outcome": None,
            "validation_status": None,
        })


def normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    for field in REQUIRED_RESULT_FIELDS:
        result.setdefault(field, None)
    return result


def classify_batch_status(
    *,
    outcome: str | None,
    review_reason: str | None,
    confidence: float | None,
    entity_count: int,
) -> str:
    if entity_count == 0:
        return "review"
    if confidence is None or confidence < 0.65:
        return "review"
    if review_reason == "accept_with_route_audit":
        return "accepted"
    if outcome in ACCEPTED_OUTCOMES:
        return "accepted"
    return "review"


def review_reasons_for(
    *,
    status: str,
    entity_count: int,
    confidence: float | None,
    confidence_breakdown: dict[str, Any] | None,
) -> list[str]:
    if status != "review":
        return []

    reasons: list[str] = []
    breakdown = confidence_breakdown if isinstance(confidence_breakdown, dict) else {}
    coverage_score = safe_float(breakdown.get("coverage"))
    diversity_score = safe_float(breakdown.get("diversity"))
    extractor_weight = safe_float(breakdown.get("extractor_weight"))

    if entity_count == 0:
        reasons.append("empty_extraction")
    if confidence is not None and confidence < 0.65:
        reasons.append("confidence_below_threshold")
    if entity_count <= 1:
        reasons.append("low_entity_count")
    if coverage_score is not None and coverage_score < 0.3:
        reasons.append("low_coverage")
    if diversity_score is not None and diversity_score < 0.3:
        reasons.append("low_diversity")
    if extractor_weight is not None and extractor_weight < 0.7:
        reasons.append("low_extractor_weight")
    return reasons


def build_initial_text_diagnostics(source_path: Path) -> dict[str, Any]:
    if source_path.suffix.lower() == ".txt":
        return analyze_text(source_path.read_text(encoding="utf-8", errors="replace"), method="plain_text")
    return analyze_text("", method=infer_pdf_method({}) if source_path.suffix.lower() == ".pdf" else "unsupported")


def diagnostics_from_result(
    source_path: Path,
    extractor_result: dict[str, Any],
    fallback_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    if source_path.suffix.lower() == ".txt":
        return fallback_diagnostics

    raw_text = str(extractor_result.get("raw_text") or "")
    method = infer_pdf_method(extractor_result)
    if raw_text:
        return analyze_text(raw_text, method=method)

    diagnostics = dict(fallback_diagnostics)
    diagnostics["method"] = method
    diagnostics["suspicious"] = True
    return diagnostics


def infer_pdf_method(extractor_result: dict[str, Any]) -> str:
    if extractor_result.get("ocr_fallback_used") or extractor_result.get("ocr_engine"):
        return "tesseract fallback"
    status = str(extractor_result.get("text_quality_status") or "").lower()
    if status in {"readable_native", "unreadable_after_ocr", "ocr_fallback_applied"}:
        if "ocr" in status:
            return "pymupdf"
        try:
            from ingestion.pdf_pipeline import DOCLING_AVAILABLE
        except Exception:
            DOCLING_AVAILABLE = False
        return "docling" if DOCLING_AVAILABLE else "pymupdf"
    return "unknown"


def analyze_text(text: str, *, method: str) -> dict[str, Any]:
    normalized = str(text or "")
    stripped = normalized.strip()
    length = len(stripped)
    compact = re.sub(r"\s+", " ", stripped)
    suspicious = (
        length < 50
        or non_alnum_ratio(stripped) > 0.40
        or bool(OCR_ARTIFACT_PATTERN.search(stripped))
    )
    return {
        "length": length,
        "preview": compact[:300],
        "method": method,
        "suspicious": bool(suspicious),
    }


def non_alnum_ratio(text: str) -> float:
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return 1.0
    non_alnum = sum(1 for char in visible if not char.isalnum())
    return non_alnum / len(visible)


def review_reason_for(result, extractor_result: dict[str, Any]) -> str | None:
    validation_errors = getattr(result, "validation_errors", None) or []
    if validation_errors:
        codes = [str(error.get("code", "validation_error")) for error in validation_errors if isinstance(error, dict)]
        return ", ".join(codes) if codes else "validation_error"
    validation_status = getattr(result, "validation_status", None)
    if validation_status and validation_status != "accepted":
        return str(validation_status)
    review_recommendation = extractor_result.get("review_recommendation")
    if review_recommendation:
        return str(review_recommendation)
    outcome = getattr(result, "outcome", None)
    if outcome and outcome not in ACCEPTED_OUTCOMES:
        return str(outcome)
    return None


def write_batch_reports(summary: dict[str, Any]) -> tuple[Path, Path]:
    ensure_batch_validation_dirs()
    BATCH_JSON_REPORT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# MedAI Batch Real-Document Validation",
        "",
        f"- Generated at: `{summary['timestamp']}`",
        f"- Input dir: `{summary['input_dir']}`",
        f"- Total files: `{summary['total_files']}`",
        f"- Accepted: `{summary['accepted_count']}`",
        f"- Review: `{summary['review_count']}`",
        f"- Errors: `{summary['error_count']}`",
        f"- Empty extractions: `{summary['empty_extraction_count']}`",
        f"- Fallbacks: `{summary['fallback_count']}`",
        f"- Low text quality: `{summary.get('low_text_quality_count', 0)}`",
        f"- Average text length: `{summary.get('avg_text_length', 0)}`",
        "",
        "## Review Reason Breakdown",
        "",
        *[
            f"- {reason}: `{summary.get('review_reason_summary', {}).get(reason, 0)}`"
            for reason in REVIEW_REASON_KEYS
        ],
        "",
        "## Results",
        "",
    ]
    if not summary["results"]:
        lines.append("No supported PDF or TXT files found in `real_validation_input/`.")
    else:
        lines.extend([
            "| File | Status | Entities | Extractor | Text length | Text method | Suspicious text | Fallback | Confidence | Why reviewed | Review reason | Error |",
            "| --- | --- | ---: | --- | ---: | --- | --- | --- | ---: | --- | --- | --- |",
        ])
        for item in summary["results"]:
            fallback = item.get("fallback_extractor") or item.get("fallback_reason") or ""
            diagnostics = item.get("text_diagnostics") or {}
            lines.append(
                "| "
                + " | ".join([
                    escape_md(item.get("filename")),
                    escape_md(item.get("status")),
                    str(item.get("entity_count", 0)),
                    escape_md(item.get("selected_extractor")),
                    str(diagnostics.get("length", 0)),
                    escape_md(diagnostics.get("method")),
                    "yes" if diagnostics.get("suspicious") else "no",
                    escape_md(fallback),
                    "" if item.get("confidence") is None else str(item.get("confidence")),
                    escape_md(", ".join(item.get("why_reviewed") or [])),
                    escape_md(item.get("review_reason")),
                    escape_md(item.get("error")),
                ])
                + " |"
            )

    lines.extend(["", "## Files Needing Review", ""])
    if summary["files_needing_review"]:
        lines.extend(f"- `{name}`" for name in summary["files_needing_review"])
    else:
        lines.append("- None")

    lines.extend(["", "## Errors", ""])
    if summary["errors"]:
        lines.extend(f"- `{item['filename']}`: {item['error']}" for item in summary["errors"])
    else:
        lines.append("- None")

    BATCH_MD_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_review_audit_reports_from_latest()
    return BATCH_JSON_REPORT, BATCH_MD_REPORT


def write_review_audit_reports_from_latest() -> tuple[Path, Path]:
    ensure_batch_validation_dirs()
    if BATCH_JSON_REPORT.exists():
        summary = json.loads(BATCH_JSON_REPORT.read_text(encoding="utf-8"))
    else:
        summary = {"timestamp": datetime.now(UTC).isoformat(), "results": []}

    audit = build_review_audit(summary)
    REVIEW_AUDIT_JSON_REPORT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    REVIEW_AUDIT_MD_REPORT.write_text(render_review_audit_markdown(audit), encoding="utf-8")
    return REVIEW_AUDIT_JSON_REPORT, REVIEW_AUDIT_MD_REPORT


def build_review_audit(summary: dict[str, Any]) -> dict[str, Any]:
    reviewed = [item for item in summary.get("results", []) if item.get("status") == "review"]
    items = [review_audit_item(item) for item in reviewed]
    breakdown = {category: 0 for category in REVIEW_FIX_CATEGORIES}
    for item in items:
        category = item["recommended_fix_category"]
        if category in breakdown:
            breakdown[category] += 1
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_report": str(BATCH_JSON_REPORT),
        "source_timestamp": summary.get("timestamp"),
        "total_reviewed": len(items),
        "review_fix_breakdown": breakdown,
        "files": items,
    }


REVIEW_FIX_CATEGORIES = (
    "no_entities",
    "low_entity_count",
    "low_confidence",
    "low_coverage",
    "low_diversity",
    "extractor_issue",
)


def review_audit_item(item: dict[str, Any]) -> dict[str, Any]:
    diagnostics = item.get("text_diagnostics") if isinstance(item.get("text_diagnostics"), dict) else {}
    breakdown = normalized_review_confidence_breakdown(item.get("confidence_breakdown"))
    entity_count = int(item.get("entity_count") or 0)
    confidence = safe_float(item.get("confidence"))
    return {
        "filename": item.get("filename"),
        "entity_count": entity_count,
        "entities": list(item.get("entities") or []),
        "confidence": confidence,
        "confidence_breakdown": breakdown,
        "why_reviewed": list(item.get("why_reviewed") or []),
        "text_preview": str(diagnostics.get("preview") or "")[:300],
        "text_length": int(diagnostics.get("length") or 0),
        "extraction_method": diagnostics.get("method"),
        "recommended_fix_category": recommended_fix_category(
            entity_count=entity_count,
            confidence=confidence,
            coverage_score=breakdown["coverage_score"],
            diversity_score=breakdown["diversity_score"],
            extractor_weight=breakdown["extractor_weight"],
        ),
    }


def normalized_review_confidence_breakdown(value: Any) -> dict[str, float | None]:
    source = value if isinstance(value, dict) else {}
    extractor_weight = safe_float(source.get("extractor_weight"))
    calibrated_weight = safe_float(source.get("calibrated_extractor_weight"))
    return {
        "entity_count_score": safe_float(source.get("entity_count")),
        "coverage_score": safe_float(source.get("coverage")),
        "diversity_score": safe_float(source.get("diversity")),
        "extractor_weight": extractor_weight,
        "calibrated_weight": calibrated_weight if calibrated_weight is not None else extractor_weight,
    }


def recommended_fix_category(
    *,
    entity_count: int,
    confidence: float | None,
    coverage_score: float | None,
    diversity_score: float | None,
    extractor_weight: float | None,
) -> str:
    if entity_count == 0:
        return "no_entities"
    if entity_count == 1:
        return "low_entity_count"
    if confidence is not None and confidence < 0.5:
        return "low_confidence"
    if coverage_score is not None and coverage_score < 0.3:
        return "low_coverage"
    if diversity_score is not None and diversity_score < 0.3:
        return "low_diversity"
    if extractor_weight is not None and extractor_weight < 0.7:
        return "extractor_issue"
    return "low_confidence"


def render_review_audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# MedAI Review Audit",
        "",
        f"- Source report: `{audit.get('source_report')}`",
        f"- Reviewed files: `{audit.get('total_reviewed', 0)}`",
        "",
        "## Review Fix Breakdown",
        "",
    ]
    breakdown = audit.get("review_fix_breakdown") or {}
    lines.extend(f"- {category}: `{breakdown.get(category, 0)}`" for category in REVIEW_FIX_CATEGORIES)

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
            f"- confidence: {item.get('confidence')}",
            f"- entities: {entity_texts}",
            f"- why reviewed: {item.get('why_reviewed') or []}",
            f"- recommended fix: {item.get('recommended_fix_category')}",
            "",
            "- preview:",
            str(item.get("text_preview") or ""),
        ])

    return "\n".join(lines) + "\n"


def copy_to_unique_destination(source_path: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = unique_destination(destination_dir / source_path.name)
    shutil.copy2(source_path, destination)
    return destination


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate unique path for {path}")


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def escape_md(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def main() -> int:
    summary = run_batch_validation()
    if summary["total_files"] == 0:
        print("No supported PDF or TXT files found in real_validation_input/. Nothing to validate.")
    else:
        print("MedAI batch validation complete.")
    print(f"total: {summary['total_files']}")
    print(f"accepted: {summary['accepted_count']}")
    print(f"review: {summary['review_count']}")
    print(f"errors: {summary['error_count']}")
    print(f"empty_extractions: {summary['empty_extraction_count']}")
    print(f"fallbacks: {summary['fallback_count']}")
    print(f"low_text_quality: {summary['low_text_quality_count']}")
    print(f"avg_text_length: {summary['avg_text_length']}")
    print(f"review_reason_summary: {summary['review_reason_summary']}")
    print(f"json_report: {BATCH_JSON_REPORT}")
    print(f"markdown_report: {BATCH_MD_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
