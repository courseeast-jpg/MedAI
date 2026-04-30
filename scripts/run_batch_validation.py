from __future__ import annotations

import json
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
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}
ACCEPTED_OUTCOMES = {"written"}


REQUIRED_RESULT_FIELDS = [
    "filename",
    "status",
    "entity_count",
    "selected_extractor",
    "primary_extractor",
    "fallback_extractor",
    "fallback_reason",
    "terminal_empty_prevented",
    "confidence",
    "review_reason",
    "error",
]


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
        else:
            summary["error_count"] += 1
            summary["errors"].append({"filename": result["filename"], "error": result["error"]})
        if int(result["entity_count"]) == 0:
            summary["empty_extraction_count"] += 1
        if result["fallback_extractor"] or result["fallback_reason"] or result["terminal_empty_prevented"]:
            summary["fallback_count"] += 1

    write_batch_reports(summary)
    return summary


def process_one_file(pipeline: ExecutionPipeline, source_path: Path, *, run_id: str) -> dict[str, Any]:
    try:
        if source_path.suffix.lower() == ".pdf":
            result = pipeline.process_pdf(source_path, specialty="general", session_id=run_id)
        elif source_path.suffix.lower() == ".txt":
            result = pipeline.process_text(
                source_path.read_text(encoding="utf-8", errors="replace"),
                specialty="general",
                source_name=source_path.name,
                session_id=run_id,
            )
        else:
            raise ValueError(f"Unsupported file type: {source_path.suffix}")

        extractor_result = dict(result.extractor_result or {})
        audit = dict(result.audit or {})
        entities = list(extractor_result.get("entities", []))
        selected_extractor = (
            extractor_result.get("selected_extractor")
            or extractor_result.get("actual_extractor")
            or audit.get("extractor_actual")
            or audit.get("extractor")
        )
        status = "accepted" if result.outcome in ACCEPTED_OUTCOMES else "review"
        copy_destination = copy_to_unique_destination(
            source_path,
            BATCH_ARCHIVE_DIR if status == "accepted" else BATCH_REVIEW_DIR,
        )
        review_reason = review_reason_for(result, extractor_result)
        return normalize_result({
            "filename": source_path.name,
            "status": status,
            "entity_count": len(entities),
            "selected_extractor": str(selected_extractor) if selected_extractor is not None else None,
            "primary_extractor": extractor_result.get("primary_extractor"),
            "fallback_extractor": extractor_result.get("fallback_extractor"),
            "fallback_reason": extractor_result.get("fallback_reason"),
            "terminal_empty_prevented": bool(extractor_result.get("terminal_empty_prevented", False)),
            "confidence": safe_float(extractor_result.get("confidence", audit.get("confidence"))),
            "review_reason": review_reason,
            "error": None,
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
            "selected_extractor": None,
            "primary_extractor": None,
            "fallback_extractor": None,
            "fallback_reason": None,
            "terminal_empty_prevented": False,
            "confidence": None,
            "review_reason": None,
            "error": str(exc),
            "copied_to": str(copy_destination),
            "outcome": None,
            "validation_status": None,
        })


def normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    for field in REQUIRED_RESULT_FIELDS:
        result.setdefault(field, None)
    return result


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
        "",
        "## Results",
        "",
    ]
    if not summary["results"]:
        lines.append("No supported PDF or TXT files found in `real_validation_input/`.")
    else:
        lines.extend([
            "| File | Status | Entities | Extractor | Fallback | Confidence | Review reason | Error |",
            "| --- | --- | ---: | --- | --- | ---: | --- | --- |",
        ])
        for item in summary["results"]:
            fallback = item.get("fallback_extractor") or item.get("fallback_reason") or ""
            lines.append(
                "| "
                + " | ".join([
                    escape_md(item.get("filename")),
                    escape_md(item.get("status")),
                    str(item.get("entity_count", 0)),
                    escape_md(item.get("selected_extractor")),
                    escape_md(fallback),
                    "" if item.get("confidence") is None else str(item.get("confidence")),
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
    return BATCH_JSON_REPORT, BATCH_MD_REPORT


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
    print(f"json_report: {BATCH_JSON_REPORT}")
    print(f"markdown_report: {BATCH_MD_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

