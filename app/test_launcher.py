"""Local MedAI test-run launcher helpers.

This module intentionally wraps the existing execution pipeline without changing
extraction, validation, routing, or MKB write behavior.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.lab_document_metadata import (
    display_document_type,
    normalize_text_quality_label,
    reason_label_for_validation,
    review_reason_for_result,
)


ROOT = Path(__file__).resolve().parent.parent
TEST_INPUT_DIR = ROOT / "test_input"
TEST_OUTPUT_DIR = ROOT / "test_output"
TEST_REVIEW_DIR = ROOT / "test_review"
TEST_ARCHIVE_DIR = ROOT / "test_archive"
TEST_RUN_REPORT_DIR = ROOT / "reports" / "test_runs"
LATEST_JSON_REPORT = TEST_RUN_REPORT_DIR / "latest_test_run.json"
LATEST_MD_REPORT = TEST_RUN_REPORT_DIR / "latest_test_run.md"

SUPPORTED_TEST_EXTENSIONS = {".pdf", ".txt"}
ACCEPTED_OUTCOMES = {"written"}
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._ -]+")


@dataclass
class TestFileResult:
    file_name: str
    status: str
    outcome: str | None = None
    processed_path: str | None = None
    selected_extractor: str | None = None
    confidence: float | None = None
    validation_status: str | None = None
    document_type: str | None = None
    ocr_quality_band: str | None = None
    language_text_visibility: str | None = None
    cyrillic_ocr_recommended: bool = False
    ocr_gate_reason: str | None = None
    ocr_gate_review_only: bool = True
    ocr_gate_auto_accept_allowed: bool = False
    ocr_gate_fallback_executed: bool = False
    operator_review_reason: str | None = None
    operator_reason_label: str | None = None
    error: str | None = None


@dataclass
class TestRunSummary:
    timestamp: str
    run_id: str
    files_attempted: list[str] = field(default_factory=list)
    files_processed: list[str] = field(default_factory=list)
    files_accepted: list[str] = field(default_factory=list)
    files_sent_to_review: list[str] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    selected_extractors: dict[str, str | None] = field(default_factory=dict)
    confidence: dict[str, float | None] = field(default_factory=dict)
    results: list[dict[str, Any]] = field(default_factory=list)
    json_report_path: str = str(LATEST_JSON_REPORT)
    markdown_report_path: str = str(LATEST_MD_REPORT)

    @property
    def accepted_count(self) -> int:
        return len(self.files_accepted)

    @property
    def review_count(self) -> int:
        return len(self.files_sent_to_review)

    @property
    def error_count(self) -> int:
        return len(self.errors)


def ensure_test_launcher_dirs(root: Path = ROOT) -> None:
    for relative in (
        "test_input",
        "test_output",
        "test_review",
        "test_archive",
        "reports/test_runs",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)


def list_test_input_files(input_dir: Path = TEST_INPUT_DIR) -> list[Path]:
    input_dir.mkdir(parents=True, exist_ok=True)
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_TEST_EXTENSIONS)


def save_uploaded_test_file(uploaded_file, input_dir: Path = TEST_INPUT_DIR) -> Path:
    input_dir.mkdir(parents=True, exist_ok=True)
    destination = _unique_destination(input_dir / safe_test_filename(uploaded_file.name))
    destination.write_bytes(_uploaded_file_bytes(uploaded_file))
    return destination


def safe_test_filename(filename: str) -> str:
    original_name = Path(filename or "").name
    safe_name = SAFE_FILENAME_RE.sub("_", original_name).strip(" .")
    if not safe_name:
        raise ValueError("Uploaded test file has no usable filename.")
    suffix = Path(safe_name).suffix.lower()
    if suffix not in SUPPORTED_TEST_EXTENSIONS:
        raise ValueError(f"Unsupported test file type: {suffix or 'none'}")
    return safe_name


def clear_test_input(input_dir: Path = TEST_INPUT_DIR) -> list[Path]:
    input_dir.mkdir(parents=True, exist_ok=True)
    removed: list[Path] = []
    for path in list_test_input_files(input_dir):
        path.unlink()
        removed.append(path)
    gitkeep = input_dir / ".gitkeep"
    gitkeep.touch(exist_ok=True)
    return removed


def remove_test_input_file(filename: str, input_dir: Path = TEST_INPUT_DIR) -> Path | None:
    input_dir.mkdir(parents=True, exist_ok=True)
    requested = Path(filename or "").name
    target = input_dir / requested
    if not target.exists() or not target.is_file() or target.suffix.lower() not in SUPPORTED_TEST_EXTENSIONS:
        return None
    target.unlink()
    (input_dir / ".gitkeep").touch(exist_ok=True)
    return target


def clear_latest_test_reports(report_dir: Path = TEST_RUN_REPORT_DIR) -> list[Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    removed: list[Path] = []
    for filename in ("latest_test_run.md", "latest_test_run.json"):
        path = report_dir / filename
        if path.exists():
            path.unlink()
            removed.append(path)
    return removed


def build_test_launcher_display_state(
    queued_files: list[Path],
    latest: dict[str, Any] | None,
    current_run: TestRunSummary | None = None,
) -> dict[str, Any]:
    queued_names = [path.name for path in queued_files]
    if current_run is not None:
        return {
            "run_status": "Current run complete",
            "counter_source": "current run",
            "queued_count": len(queued_names),
            "queued_files": queued_names,
            "accepted": current_run.accepted_count,
            "review": current_run.review_count,
            "errors": current_run.error_count,
            "latest_report_timestamp": current_run.timestamp,
            "show_no_supported_files": not queued_names,
        }
    if not queued_names:
        return {
            "run_status": "No current run",
            "counter_source": "no current run",
            "queued_count": 0,
            "queued_files": [],
            "accepted": 0,
            "review": 0,
            "errors": 0,
            "latest_report_timestamp": (latest or {}).get("timestamp"),
            "show_no_supported_files": True,
        }
    return {
        "run_status": "Queued files ready",
        "counter_source": "previous report" if latest else "no previous report",
        "queued_count": len(queued_names),
        "queued_files": queued_names,
        "accepted": int((latest or {}).get("accepted_count", 0)),
        "review": int((latest or {}).get("review_count", 0)),
        "errors": int((latest or {}).get("error_count", 0)),
        "latest_report_timestamp": (latest or {}).get("timestamp"),
        "show_no_supported_files": False,
    }


def run_medai_test_batch(execution_pipeline, *, specialty: str = "general") -> TestRunSummary:
    ensure_test_launcher_dirs()
    run_id = str(uuid4())
    summary = TestRunSummary(timestamp=datetime.now(UTC).isoformat(), run_id=run_id)

    for source_path in list_test_input_files():
        summary.files_attempted.append(source_path.name)
        file_result = _process_one_file(execution_pipeline, source_path, specialty=specialty, run_id=run_id)
        summary.results.append(file_result.__dict__)
        if file_result.status == "accepted":
            summary.files_processed.append(source_path.name)
            summary.files_accepted.append(source_path.name)
        elif file_result.status == "review":
            summary.files_processed.append(source_path.name)
            summary.files_sent_to_review.append(source_path.name)
        else:
            summary.errors.append({"file": source_path.name, "error": file_result.error or "unknown error"})
        summary.selected_extractors[source_path.name] = file_result.selected_extractor
        summary.confidence[source_path.name] = file_result.confidence

    write_test_run_reports(summary)
    return summary


def write_test_run_reports(summary: TestRunSummary) -> tuple[Path, Path]:
    ensure_test_launcher_dirs()
    data = {
        "timestamp": summary.timestamp,
        "run_id": summary.run_id,
        "files_attempted": summary.files_attempted,
        "files_processed": summary.files_processed,
        "files_accepted": summary.files_accepted,
        "files_sent_to_review": summary.files_sent_to_review,
        "errors": summary.errors,
        "selected_extractor": summary.selected_extractors,
        "confidence": summary.confidence,
        "accepted_count": summary.accepted_count,
        "review_count": summary.review_count,
        "error_count": summary.error_count,
        "results": summary.results,
    }
    LATEST_JSON_REPORT.write_text(json.dumps(data, indent=2), encoding="utf-8")

    lines = [
        "# MedAI Local Test Run",
        "",
        f"- Timestamp: `{summary.timestamp}`",
        f"- Run ID: `{summary.run_id}`",
        f"- Files attempted: `{len(summary.files_attempted)}`",
        f"- Files processed: `{len(summary.files_processed)}`",
        f"- Files accepted: `{summary.accepted_count}`",
        f"- Files sent to review: `{summary.review_count}`",
        f"- Errors: `{summary.error_count}`",
        "",
        "## Results",
        "",
    ]
    if not summary.results:
        lines.append("- No supported files found in `test_input/`.")
    else:
        for result in summary.results:
            lines.append(
                f"- `{result['file_name']}` status=`{result['status']}` "
                f"outcome=`{result.get('outcome')}` extractor=`{result.get('selected_extractor')}` "
                f"confidence=`{result.get('confidence')}`"
            )
    LATEST_MD_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return LATEST_JSON_REPORT, LATEST_MD_REPORT


def load_latest_test_run() -> dict[str, Any] | None:
    if not LATEST_JSON_REPORT.exists():
        return None
    return json.loads(LATEST_JSON_REPORT.read_text(encoding="utf-8"))


def _process_one_file(execution_pipeline, source_path: Path, *, specialty: str, run_id: str) -> TestFileResult:
    try:
        if source_path.suffix.lower() == ".pdf":
            result = execution_pipeline.process_pdf(source_path, specialty=specialty, session_id=run_id)
        elif source_path.suffix.lower() == ".txt":
            result = execution_pipeline.process_text(
                source_path.read_text(encoding="utf-8", errors="replace"),
                specialty=specialty,
                source_name=source_path.name,
                session_id=run_id,
            )
        else:
            raise ValueError(f"Unsupported test file type: {source_path.suffix}")

        extractor_result = dict(result.extractor_result or {})
        audit = dict(result.audit or {})
        selected_extractor = (
            extractor_result.get("selected_extractor")
            or extractor_result.get("actual_extractor")
            or audit.get("extractor_actual")
            or audit.get("extractor")
        )
        confidence = _safe_float(extractor_result.get("confidence", audit.get("confidence")))
        accepted = result.outcome in ACCEPTED_OUTCOMES
        validation_reason_codes = _validation_reason_codes(result.validation_errors)
        document_type = display_document_type(
            audit.get("document_type") or extractor_result.get("document_type"),
            text=str(extractor_result.get("raw_text") or extractor_result.get("text") or ""),
        )
        ocr_quality = normalize_text_quality_label(
            audit.get("ocr_quality_band"),
            audit.get("input_quality_band"),
            extractor_result.get("ocr_quality_band"),
            extractor_result.get("input_quality_band"),
            extractor_result.get("text_quality_status"),
            audit.get("text_quality_status"),
        )
        operator_reason = review_reason_for_result(
            document_type=document_type,
            validation_status=result.validation_status,
            confidence=confidence,
            status="accepted" if accepted else "review",
        )
        operator_reason_label = reason_label_for_validation(result.validation_status, validation_reason_codes)
        destination_dir = TEST_ARCHIVE_DIR if accepted else TEST_REVIEW_DIR
        destination = _move_to_unique_destination(source_path, destination_dir)
        return TestFileResult(
            file_name=source_path.name,
            status="accepted" if accepted else "review",
            outcome=result.outcome,
            processed_path=str(destination),
            selected_extractor=str(selected_extractor) if selected_extractor is not None else None,
            confidence=confidence,
            validation_status=result.validation_status,
            document_type=document_type,
            ocr_quality_band=ocr_quality,
            language_text_visibility=extractor_result.get("language_text_visibility"),
            cyrillic_ocr_recommended=bool(extractor_result.get("cyrillic_ocr_recommended", False)),
            ocr_gate_reason=extractor_result.get("ocr_gate_reason"),
            ocr_gate_review_only=bool(extractor_result.get("ocr_gate_review_only", True)),
            ocr_gate_auto_accept_allowed=bool(extractor_result.get("ocr_gate_auto_accept_allowed", False)),
            ocr_gate_fallback_executed=bool(extractor_result.get("ocr_gate_fallback_executed", False)),
            operator_review_reason=operator_reason,
            operator_reason_label=operator_reason_label,
        )
    except Exception as exc:
        destination = _move_to_unique_destination(source_path, TEST_REVIEW_DIR)
        return TestFileResult(
            file_name=source_path.name,
            status="error",
            processed_path=str(destination),
            error=str(exc),
        )


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _validation_reason_codes(errors: Any) -> list[str]:
    if not isinstance(errors, list):
        return []
    return [str(item.get("code")) for item in errors if isinstance(item, dict) and item.get("code")]


def _move_to_unique_destination(source_path: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = _unique_destination(destination_dir / source_path.name)
    return shutil.move(str(source_path), str(destination)) and destination


def _unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(1, 10_000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate unique path for {path}")


def _uploaded_file_bytes(uploaded_file) -> bytes:
    if hasattr(uploaded_file, "getbuffer"):
        return bytes(uploaded_file.getbuffer())
    if hasattr(uploaded_file, "read"):
        return bytes(uploaded_file.read())
    raise TypeError("Uploaded file object does not expose getbuffer() or read().")
