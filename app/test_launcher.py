"""Local MedAI test-run launcher helpers.

This module intentionally wraps the existing execution pipeline without changing
extraction, validation, routing, or MKB write behavior.
"""

from __future__ import annotations

import json
import inspect
import re
import shutil as shutil_module
import shutil
import subprocess
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.util import find_spec
from pathlib import Path
from typing import Any
from xml.etree import ElementTree
from uuid import uuid4

from app.lab_document_metadata import (
    UNKNOWN_DOCUMENT_LABEL,
    display_document_type,
    normalize_text_quality_label,
    reason_label_for_validation,
    review_reason_for_result,
)
from ingestion.cyrillic_ocr_gate import build_cyrillic_ocr_shadow_marker


ROOT = Path(__file__).resolve().parent.parent
TEST_INPUT_DIR = ROOT / "test_input"
TEST_OUTPUT_DIR = ROOT / "test_output"
TEST_REVIEW_DIR = ROOT / "test_review"
TEST_ARCHIVE_DIR = ROOT / "test_archive"
TEST_RUN_REPORT_DIR = ROOT / "reports" / "test_runs"
LATEST_JSON_REPORT = TEST_RUN_REPORT_DIR / "latest_test_run.json"
LATEST_MD_REPORT = TEST_RUN_REPORT_DIR / "latest_test_run.md"

SUPPORTED_TEST_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".docx",
}
RUN_REVIEW_UPLOAD_TYPES = tuple(extension.lstrip(".") for extension in sorted(SUPPORTED_TEST_EXTENSIONS))
IMAGE_TEST_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
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
    ocr_gate_fallback_engine: str | None = None
    ocr_gate_fallback_language: str | None = None
    ocr_gate_fallback_cyrillic_detected: bool = False
    ocr_gate_fallback_text_visibility: str | None = None
    ocr_gate_fallback_review_only: bool = True
    ocr_gate_fallback_auto_accept_allowed: bool = False
    ocr_gate_fallback_error_bucket: str | None = None
    ocr_gate_fallback_classification_diagnostic: dict | None = None
    ocr_gate_fallback_treatment_classification_diagnostic: dict | None = None
    document_family_classification_diagnostic: dict | None = None
    operator_review_reason: str | None = None
    operator_reason_label: str | None = None
    # MEDAI-CORPUS-EXTRACTION-TO-MKB-MINIMUM-01: hand-off fields for the
    # Run & Review "Extracted information preview" section. All fields are
    # public-safe and carry no raw OCR text, raw lines, filenames, or PHI.
    extracted_medical_facts_preview_safe: list[dict] = field(default_factory=list)
    extracted_medical_fact_count: int = 0
    extracted_medical_fact_types: list[str] = field(default_factory=list)
    extraction_to_mkb_candidate_count: int = 0
    extraction_to_mkb_written_count: int = 0
    extraction_to_mkb_review_count: int = 0
    # MEDAI-CORPUS-EXTRACTION-TO-MKB-MINIMUM-03: per-row MKB record IDs and
    # states so the UI can render operator review action affordances next to
    # each preview row. Public-safe — no PHI / raw text / private paths.
    extracted_medical_fact_record_ids: list[str] = field(default_factory=list)
    extracted_medical_fact_record_states: dict = field(default_factory=dict)
    extractor_dispatch_count: int = 0
    extraction_candidates_count: int = 0
    candidates_after_filter_count: int = 0
    records_written_count: int = 0
    records_deduped_count: int = 0
    review_bound_records_written_count: int = 0
    source_modality: str | None = None
    run_review_summary: str | None = None
    files_processed: int = 1
    ocr_attempted: bool = False
    ocr_available: bool = False
    ocr_recovered: bool = False
    ocr_text_character_bucket: str | None = None
    ocr_line_count_bucket: str | None = None
    ocr_has_table_like_layout: bool = False
    ocr_has_key_value_like_layout: bool = False
    ocr_has_section_heading_like_layout: bool = False
    extractor_dispatch_attempted: bool = False
    extractor_dispatch_family: str | None = None
    extraction_candidates_created: int = 0
    extraction_candidates_after_filter: int = 0
    extraction_candidates_dropped: int = 0
    drop_reason_counts: dict[str, int] = field(default_factory=dict)
    records_written: int = 0
    records_deduped: int = 0
    review_bound_records_written: int = 0
    selected_document_category: str | None = None
    selected_specialty_domain: str | None = None
    document_type_before_extraction: str | None = None
    document_type_after_extraction: str | None = None
    runtime_diagnostic_summary: str | None = None
    image_ocr_available: bool = False
    image_ocr_attempted: bool = False
    image_ocr_engine: str | None = None
    image_ocr_text_visibility: str | None = None
    image_ocr_review_only: bool = True
    image_ocr_auto_accept_allowed: bool = False
    external_api_used: bool = False
    error: str | None = None


@dataclass(frozen=True)
class LocalImageOcrResult:
    available: bool
    attempted: bool
    text: str = ""
    engine: str | None = None
    language: str | None = None
    text_visibility: str = "unavailable"
    error_bucket: str | None = None


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

    @property
    def safe_ocr_pipeline_diagnostics(self) -> dict[str, Any]:
        return build_safe_ocr_pipeline_diagnostics(self)

    @property
    def safe_real_run_extraction_diagnostics(self) -> dict[str, Any]:
        return build_safe_real_run_extraction_diagnostics(self)


def build_safe_ocr_pipeline_diagnostics(summary: TestRunSummary) -> dict[str, Any]:
    """Return count-only OCR extraction diagnostics for reports/UI.

    No raw OCR text, raw filenames, private paths, or uploaded names are
    included. Per-file entries use generated ordinal IDs only.
    """
    per_file_document_type: dict[str, str] = {}
    for index, item in enumerate(summary.results, start=1):
        safe_id = f"file_{index:03d}"
        per_file_document_type[safe_id] = str(item.get("document_type") or UNKNOWN_DOCUMENT_LABEL)
    return {
        "files_processed": int(len(summary.files_processed)),
        "ocr_attempted_count": sum(1 for item in summary.results if item.get("image_ocr_attempted")),
        "ocr_recovered_count": sum(
            1 for item in summary.results if item.get("image_ocr_text_visibility") == "recovered"
        ),
        "extractor_dispatch_count": sum(int(item.get("extractor_dispatch_count") or 0) for item in summary.results),
        "extraction_candidates_count": sum(int(item.get("extraction_candidates_count") or 0) for item in summary.results),
        "candidates_after_filter_count": sum(int(item.get("candidates_after_filter_count") or 0) for item in summary.results),
        "records_written_count": sum(int(item.get("records_written_count") or 0) for item in summary.results),
        "records_deduped_count": sum(int(item.get("records_deduped_count") or 0) for item in summary.results),
        "review_bound_records_written_count": sum(
            int(item.get("review_bound_records_written_count") or 0) for item in summary.results
        ),
        "per_file_document_type": per_file_document_type,
    }


def build_safe_real_run_extraction_diagnostics(summary: TestRunSummary) -> dict[str, Any]:
    """Return 14C runtime counters without raw text, names, or paths."""
    per_file: dict[str, dict[str, Any]] = {}
    drop_reason_counts: dict[str, int] = {}
    for index, item in enumerate(summary.results, start=1):
        safe_id = f"file_{index:03d}"
        reasons = dict(item.get("drop_reason_counts") or {})
        for reason, count in reasons.items():
            drop_reason_counts[str(reason)] = drop_reason_counts.get(str(reason), 0) + int(count or 0)
        per_file[safe_id] = {
            "ocr_attempted": bool(item.get("ocr_attempted")),
            "ocr_available": bool(item.get("ocr_available")),
            "ocr_recovered": bool(item.get("ocr_recovered")),
            "ocr_text_character_bucket": item.get("ocr_text_character_bucket"),
            "ocr_line_count_bucket": item.get("ocr_line_count_bucket"),
            "ocr_has_table_like_layout": bool(item.get("ocr_has_table_like_layout")),
            "ocr_has_key_value_like_layout": bool(item.get("ocr_has_key_value_like_layout")),
            "ocr_has_section_heading_like_layout": bool(item.get("ocr_has_section_heading_like_layout")),
            "extractor_dispatch_attempted": bool(item.get("extractor_dispatch_attempted")),
            "extractor_dispatch_family": item.get("extractor_dispatch_family"),
            "extraction_candidates_created": int(item.get("extraction_candidates_created") or 0),
            "extraction_candidates_after_filter": int(item.get("extraction_candidates_after_filter") or 0),
            "extraction_candidates_dropped": int(item.get("extraction_candidates_dropped") or 0),
            "drop_reason_counts": reasons,
            "records_written": int(item.get("records_written") or 0),
            "records_deduped": int(item.get("records_deduped") or 0),
            "review_bound_records_written": int(item.get("review_bound_records_written") or 0),
            "selected_document_category": item.get("selected_document_category"),
            "selected_specialty_domain": item.get("selected_specialty_domain"),
            "source_modality": item.get("source_modality"),
            "document_type_before_extraction": item.get("document_type_before_extraction"),
            "document_type_after_extraction": item.get("document_type_after_extraction"),
        }
    return {
        "files_processed": int(len(summary.files_processed)),
        "ocr_attempted": sum(1 for item in summary.results if item.get("ocr_attempted")),
        "ocr_available": sum(1 for item in summary.results if item.get("ocr_available")),
        "ocr_recovered": sum(1 for item in summary.results if item.get("ocr_recovered")),
        "extractor_dispatch_attempted": sum(1 for item in summary.results if item.get("extractor_dispatch_attempted")),
        "extraction_candidates_created": sum(int(item.get("extraction_candidates_created") or 0) for item in summary.results),
        "extraction_candidates_after_filter": sum(
            int(item.get("extraction_candidates_after_filter") or 0) for item in summary.results
        ),
        "extraction_candidates_dropped": sum(
            int(item.get("extraction_candidates_dropped") or 0) for item in summary.results
        ),
        "drop_reason_counts": drop_reason_counts,
        "records_written": sum(int(item.get("records_written") or 0) for item in summary.results),
        "records_deduped": sum(int(item.get("records_deduped") or 0) for item in summary.results),
        "review_bound_records_written": sum(
            int(item.get("review_bound_records_written") or 0) for item in summary.results
        ),
        "per_file": per_file,
    }


def ensure_test_launcher_dirs(root: Path = ROOT) -> None:
    for relative in (
        "test_input",
        "test_output",
        "test_review",
        "test_archive",
        "reports/test_runs",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)


def list_test_input_files(input_dir: Path | None = None) -> list[Path]:
    input_dir = input_dir or TEST_INPUT_DIR
    input_dir.mkdir(parents=True, exist_ok=True)
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_TEST_EXTENSIONS)


def save_uploaded_test_file(uploaded_file, input_dir: Path | None = None) -> Path:
    input_dir = input_dir or TEST_INPUT_DIR
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


def clear_test_input(input_dir: Path | None = None) -> list[Path]:
    input_dir = input_dir or TEST_INPUT_DIR
    input_dir.mkdir(parents=True, exist_ok=True)
    removed: list[Path] = []
    for path in list_test_input_files(input_dir):
        path.unlink()
        removed.append(path)
    gitkeep = input_dir / ".gitkeep"
    gitkeep.touch(exist_ok=True)
    return removed


def remove_test_input_file(filename: str, input_dir: Path | None = None) -> Path | None:
    input_dir = input_dir or TEST_INPUT_DIR
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


def run_medai_test_batch(
    execution_pipeline,
    *,
    specialty: str = "general",
    document_category: str = "",
) -> TestRunSummary:
    ensure_test_launcher_dirs()
    run_id = str(uuid4())
    summary = TestRunSummary(timestamp=datetime.now(UTC).isoformat(), run_id=run_id)

    for source_path in list_test_input_files():
        summary.files_attempted.append(source_path.name)
        file_result = _process_one_file(
            execution_pipeline,
            source_path,
            specialty=specialty,
            run_id=run_id,
            document_category=document_category,
        )
        summary.results.append(file_result.__dict__)
        if file_result.status == "accepted":
            summary.files_processed.append(source_path.name)
            summary.files_accepted.append(source_path.name)
        elif file_result.status in {"review", "review_ocr_quality"}:
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
    LATEST_JSON_REPORT.parent.mkdir(parents=True, exist_ok=True)
    LATEST_MD_REPORT.parent.mkdir(parents=True, exist_ok=True)
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
        "safe_ocr_pipeline_diagnostics": summary.safe_ocr_pipeline_diagnostics,
        "safe_real_run_extraction_diagnostics": summary.safe_real_run_extraction_diagnostics,
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


def runtime_cyrillic_ocr_marker_for_result(extractor_result: dict[str, Any]) -> dict[str, Any]:
    existing_visibility = extractor_result.get("language_text_visibility")
    existing_reason = extractor_result.get("ocr_gate_reason")
    if existing_visibility and existing_reason:
        return {
            "language_text_visibility": existing_visibility,
            "cyrillic_ocr_recommended": bool(extractor_result.get("cyrillic_ocr_recommended", False)),
            "ocr_gate_reason": existing_reason,
            "ocr_gate_review_only": bool(extractor_result.get("ocr_gate_review_only", True)),
            "ocr_gate_auto_accept_allowed": bool(extractor_result.get("ocr_gate_auto_accept_allowed", False)),
            "ocr_gate_fallback_executed": bool(extractor_result.get("ocr_gate_fallback_executed", False)),
            "ocr_gate_fallback_engine": extractor_result.get("ocr_gate_fallback_engine"),
            "ocr_gate_fallback_language": extractor_result.get("ocr_gate_fallback_language"),
            "ocr_gate_fallback_cyrillic_detected": bool(
                extractor_result.get("ocr_gate_fallback_cyrillic_detected", False)
            ),
            "ocr_gate_fallback_text_visibility": extractor_result.get("ocr_gate_fallback_text_visibility"),
            "ocr_gate_fallback_review_only": bool(extractor_result.get("ocr_gate_fallback_review_only", True)),
            "ocr_gate_fallback_auto_accept_allowed": bool(
                extractor_result.get("ocr_gate_fallback_auto_accept_allowed", False)
            ),
            "ocr_gate_fallback_error_bucket": extractor_result.get("ocr_gate_fallback_error_bucket"),
            "ocr_gate_fallback_classification_diagnostic": extractor_result.get(
                "ocr_gate_fallback_classification_diagnostic"
            ),
            "document_family_classification_diagnostic": _runtime_family_diagnostic(extractor_result),
            "ocr_gate_fallback_treatment_classification_diagnostic": extractor_result.get(
                "ocr_gate_fallback_treatment_classification_diagnostic"
            ),
        }

    marker = build_cyrillic_ocr_shadow_marker(
        str(extractor_result.get("raw_text") or extractor_result.get("text") or ""),
        current_ocr_skipped=not bool(extractor_result.get("ocr_fallback_used", False)),
        language_context="unknown",
    )
    return {
        "language_text_visibility": marker.get("language_text_visibility"),
        "cyrillic_ocr_recommended": bool(marker.get("cyrillic_ocr_recommended", False)),
        "ocr_gate_reason": marker.get("ocr_gate_reason"),
        "ocr_gate_review_only": bool(marker.get("review_only", True)),
        "ocr_gate_auto_accept_allowed": bool(marker.get("auto_accept_allowed", False)),
        "ocr_gate_fallback_executed": bool(marker.get("ocr_fallback_executed", False)),
        "ocr_gate_fallback_engine": None,
        "ocr_gate_fallback_language": None,
        "ocr_gate_fallback_cyrillic_detected": False,
        "ocr_gate_fallback_text_visibility": None,
        "ocr_gate_fallback_review_only": True,
        "ocr_gate_fallback_auto_accept_allowed": False,
        "ocr_gate_fallback_error_bucket": None,
        "ocr_gate_fallback_classification_diagnostic": None,
        "ocr_gate_fallback_treatment_classification_diagnostic": None,
        "document_family_classification_diagnostic": None,
    }


def _process_one_file(
    execution_pipeline,
    source_path: Path,
    *,
    specialty: str,
    run_id: str,
    document_category: str = "",
) -> TestFileResult:
    try:
        suffix = source_path.suffix.lower()
        source_modality = ""
        image_ocr = LocalImageOcrResult(available=False, attempted=False)
        ocr_shape = _ocr_text_shape_diagnostics("")
        if suffix == ".pdf":
            result = execution_pipeline.process_pdf(source_path, specialty=specialty, session_id=run_id)
        elif suffix == ".txt":
            result = _process_text_with_optional_context(
                execution_pipeline,
                source_path.read_text(encoding="utf-8", errors="replace"),
                specialty=specialty,
                source_name=source_path.name,
                session_id=run_id,
            )
        elif suffix == ".docx":
            result = _process_text_with_optional_context(
                execution_pipeline,
                extract_docx_text_local(source_path),
                specialty=specialty,
                source_name=source_path.name,
                session_id=run_id,
            )
        elif suffix in IMAGE_TEST_EXTENSIONS:
            image_ocr = recover_text_from_image_local(source_path)
            source_modality = "image_ocr"
            if not image_ocr.available:
                return _review_bound_unavailable_file_result(
                    source_path,
                    selected_extractor="local_image_ocr_unavailable",
                    error="Local image OCR extractor is safely unavailable in this runtime.",
                    image_ocr=image_ocr,
                    specialty=specialty,
                    document_category=document_category,
                )
            if not image_ocr.text.strip():
                return _review_bound_no_text_file_result(
                    source_path,
                    image_ocr=image_ocr,
                    specialty=specialty,
                    document_category=document_category,
                )
            ocr_shape = _ocr_text_shape_diagnostics(image_ocr.text)
            result = _process_text_with_optional_context(
                execution_pipeline,
                image_ocr.text,
                specialty=specialty,
                source_name=source_path.name,
                session_id=run_id,
                source_modality="image_ocr",
                document_category=document_category,
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
        force_review_bound = suffix == ".docx" or suffix in IMAGE_TEST_EXTENSIONS
        accepted = result.outcome in ACCEPTED_OUTCOMES and not force_review_bound
        outcome = "queued_for_review" if force_review_bound else result.outcome
        validation_status = (
            "needs_review" if force_review_bound and result.validation_status == "accepted" else result.validation_status
        )
        validation_reason_codes = _validation_reason_codes(result.validation_errors)
        ocr_gate_marker = runtime_cyrillic_ocr_marker_for_result(extractor_result)
        if suffix in IMAGE_TEST_EXTENSIONS:
            ocr_gate_marker.update(_image_ocr_gate_marker(image_ocr))
        document_type_before_extraction = display_document_type(
            _runtime_document_type_candidate(audit, extractor_result, ocr_gate_marker),
            text=str(extractor_result.get("raw_text") or extractor_result.get("text") or ""),
        )
        document_type = document_type_before_extraction
        if document_type == UNKNOWN_DOCUMENT_LABEL:
            extracted_candidate_type = _document_type_from_extracted_candidates(extractor_result)
            if extracted_candidate_type:
                document_type = display_document_type(extracted_candidate_type)
        dispatch_count = int(extractor_result.get("cross_domain_extractor_dispatch_count") or 0)
        candidates_created = int(extractor_result.get("cross_domain_extraction_candidates_count") or 0)
        candidates_after_filter = int(extractor_result.get("cross_domain_candidates_after_filter_count") or 0)
        records_written = int(extractor_result.get("cross_domain_records_written_count") or 0)
        records_deduped = int(extractor_result.get("cross_domain_records_deduped_count") or 0)
        review_bound_records_written = int(
            extractor_result.get("cross_domain_review_bound_records_written_count") or 0
        )
        candidates_dropped = max(0, candidates_created - candidates_after_filter)
        drop_reason_counts = _drop_reason_counts(
            candidates_created=candidates_created,
            candidates_after_filter=candidates_after_filter,
            records_deduped=records_deduped,
            ocr_recovered=bool(suffix in IMAGE_TEST_EXTENSIONS and image_ocr.text_visibility == "recovered"),
            extractor_dispatch_attempted=dispatch_count > 0,
        )
        runtime_summary = _runtime_diagnostic_summary(
            ocr_recovered=bool(suffix in IMAGE_TEST_EXTENSIONS and image_ocr.text_visibility == "recovered"),
            extractor_dispatch_attempted=dispatch_count > 0,
            candidates_created=candidates_created,
            records_written=records_written,
            records_deduped=records_deduped,
            review_bound_records_written=review_bound_records_written,
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
            validation_status=validation_status,
            confidence=confidence,
            status="accepted" if accepted else "review",
        )
        operator_reason_label = reason_label_for_validation(validation_status, validation_reason_codes)
        destination_dir = TEST_ARCHIVE_DIR if accepted else TEST_REVIEW_DIR
        destination = _move_to_unique_destination(source_path, destination_dir)
        return TestFileResult(
            file_name=source_path.name,
            status="accepted" if accepted else "review",
            outcome=outcome,
            processed_path=str(destination),
            selected_extractor=(
                f"local_image_ocr:{selected_extractor}"
                if suffix in IMAGE_TEST_EXTENSIONS and selected_extractor is not None
                else str(selected_extractor) if selected_extractor is not None else None
            ),
            confidence=confidence,
            validation_status=validation_status,
            document_type=document_type,
            ocr_quality_band=ocr_quality,
            language_text_visibility=ocr_gate_marker["language_text_visibility"],
            cyrillic_ocr_recommended=ocr_gate_marker["cyrillic_ocr_recommended"],
            ocr_gate_reason=ocr_gate_marker["ocr_gate_reason"],
            ocr_gate_review_only=ocr_gate_marker["ocr_gate_review_only"],
            ocr_gate_auto_accept_allowed=ocr_gate_marker["ocr_gate_auto_accept_allowed"],
            ocr_gate_fallback_executed=ocr_gate_marker["ocr_gate_fallback_executed"],
            ocr_gate_fallback_engine=ocr_gate_marker["ocr_gate_fallback_engine"],
            ocr_gate_fallback_language=ocr_gate_marker["ocr_gate_fallback_language"],
            ocr_gate_fallback_cyrillic_detected=ocr_gate_marker["ocr_gate_fallback_cyrillic_detected"],
            ocr_gate_fallback_text_visibility=ocr_gate_marker["ocr_gate_fallback_text_visibility"],
            ocr_gate_fallback_review_only=ocr_gate_marker["ocr_gate_fallback_review_only"],
            ocr_gate_fallback_auto_accept_allowed=ocr_gate_marker["ocr_gate_fallback_auto_accept_allowed"],
            ocr_gate_fallback_error_bucket=ocr_gate_marker["ocr_gate_fallback_error_bucket"],
            ocr_gate_fallback_classification_diagnostic=ocr_gate_marker[
                "ocr_gate_fallback_classification_diagnostic"
            ],
            ocr_gate_fallback_treatment_classification_diagnostic=ocr_gate_marker[
                "ocr_gate_fallback_treatment_classification_diagnostic"
            ],
            document_family_classification_diagnostic=ocr_gate_marker["document_family_classification_diagnostic"],
            operator_review_reason=operator_reason,
            operator_reason_label=operator_reason_label,
            extracted_medical_facts_preview_safe=list(
                extractor_result.get("extracted_medical_facts_preview_safe") or []
            ),
            extracted_medical_fact_count=int(
                extractor_result.get("extracted_medical_fact_count") or 0
            ),
            extracted_medical_fact_types=list(
                extractor_result.get("extracted_medical_fact_types") or []
            ),
            extraction_to_mkb_candidate_count=int(
                extractor_result.get("extraction_to_mkb_candidate_count") or 0
            ),
            extraction_to_mkb_written_count=int(
                extractor_result.get("extraction_to_mkb_written_count") or 0
            ),
            extraction_to_mkb_review_count=int(
                extractor_result.get("extraction_to_mkb_review_count") or 0
            ),
            extracted_medical_fact_record_ids=list(
                extractor_result.get("extracted_medical_fact_record_ids") or []
            ),
            extracted_medical_fact_record_states=dict(
                extractor_result.get("extracted_medical_fact_record_states") or {}
            ),
            extractor_dispatch_count=dispatch_count,
            extraction_candidates_count=candidates_created,
            candidates_after_filter_count=candidates_after_filter,
            records_written_count=records_written,
            records_deduped_count=records_deduped,
            review_bound_records_written_count=review_bound_records_written,
            source_modality=str(extractor_result.get("source_modality") or source_modality or ""),
            run_review_summary=(
                _run_review_summary_from_extraction(extractor_result)
                if str(extractor_result.get("source_modality") or source_modality or "") == "image_ocr"
                else None
            ),
            files_processed=1,
            ocr_attempted=bool(suffix in IMAGE_TEST_EXTENSIONS and image_ocr.attempted),
            ocr_available=bool(suffix in IMAGE_TEST_EXTENSIONS and image_ocr.available),
            ocr_recovered=bool(suffix in IMAGE_TEST_EXTENSIONS and image_ocr.text_visibility == "recovered"),
            ocr_text_character_bucket=ocr_shape["ocr_text_character_bucket"],
            ocr_line_count_bucket=ocr_shape["ocr_line_count_bucket"],
            ocr_has_table_like_layout=ocr_shape["ocr_has_table_like_layout"],
            ocr_has_key_value_like_layout=ocr_shape["ocr_has_key_value_like_layout"],
            ocr_has_section_heading_like_layout=ocr_shape["ocr_has_section_heading_like_layout"],
            extractor_dispatch_attempted=dispatch_count > 0,
            extractor_dispatch_family=document_type_before_extraction,
            extraction_candidates_created=candidates_created,
            extraction_candidates_after_filter=candidates_after_filter,
            extraction_candidates_dropped=candidates_dropped,
            drop_reason_counts=drop_reason_counts,
            records_written=records_written,
            records_deduped=records_deduped,
            review_bound_records_written=review_bound_records_written,
            selected_document_category=document_category,
            selected_specialty_domain=specialty,
            document_type_before_extraction=document_type_before_extraction,
            document_type_after_extraction=document_type,
            runtime_diagnostic_summary=runtime_summary,
            image_ocr_available=bool(suffix in IMAGE_TEST_EXTENSIONS and image_ocr.available),
            image_ocr_attempted=bool(suffix in IMAGE_TEST_EXTENSIONS and image_ocr.attempted),
            image_ocr_engine=image_ocr.engine if suffix in IMAGE_TEST_EXTENSIONS else None,
            image_ocr_text_visibility=image_ocr.text_visibility if suffix in IMAGE_TEST_EXTENSIONS else None,
            image_ocr_review_only=True,
            image_ocr_auto_accept_allowed=False,
            external_api_used=bool(extractor_result.get("external_api_used", False)),
        )
    except Exception as exc:
        destination = _move_to_unique_destination(source_path, TEST_REVIEW_DIR)
        return TestFileResult(
            file_name=source_path.name,
            status="error",
            processed_path=str(destination),
            error=str(exc),
        )


def _process_text_with_optional_context(
    execution_pipeline,
    text: str,
    *,
    specialty: str,
    source_name: str,
    session_id: str,
    source_modality: str = "",
    document_category: str = "",
):
    kwargs = {
        "specialty": specialty,
        "source_name": source_name,
        "session_id": session_id,
    }
    if source_modality:
        kwargs["source_modality"] = source_modality
    if document_category:
        kwargs["document_category"] = document_category
    signature = inspect.signature(execution_pipeline.process_text)
    supported = set(signature.parameters)
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in supported}
    return execution_pipeline.process_text(text, **filtered_kwargs)


def extract_docx_text_local(source_path: Path) -> str:
    """Extract visible DOCX text locally with stdlib XML parsing."""
    with zipfile.ZipFile(source_path) as archive:
        try:
            document_xml = archive.read("word/document.xml")
        except KeyError as exc:
            raise ValueError("DOCX document.xml not found") from exc

    root = ElementTree.fromstring(document_xml)
    namespaces = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    fragments = [
        node.text.strip()
        for node in root.findall(".//w:t", namespaces)
        if node.text and node.text.strip()
    ]
    extracted = " ".join(fragments).strip()
    if not extracted:
        raise ValueError("DOCX contains no extractable text")
    return extracted


def local_image_ocr_available() -> bool:
    return bool(find_spec("pytesseract") and find_spec("PIL") and shutil_module.which("tesseract"))


def recover_text_from_image_local(source_path: Path, *, timeout_seconds: int = 30) -> LocalImageOcrResult:
    if not local_image_ocr_available():
        return LocalImageOcrResult(
            available=False,
            attempted=False,
            engine="tesseract_local",
            text_visibility="unavailable",
            error_bucket="local_ocr_unavailable",
        )
    try:
        from PIL import Image, ImageOps, ImageSequence
        import pytesseract
    except Exception:
        return LocalImageOcrResult(
            available=False,
            attempted=False,
            engine="tesseract_local",
            text_visibility="unavailable",
            error_bucket="local_ocr_import_failed",
        )

    language = _choose_local_image_ocr_language()
    try:
        pages: list[str] = []
        with Image.open(source_path) as image:
            for frame in ImageSequence.Iterator(image):
                normalized = ImageOps.grayscale(ImageOps.exif_transpose(frame.copy()))
                pages.append(
                    pytesseract.image_to_string(
                        normalized,
                        lang=language,
                        config="--psm 6",
                        timeout=timeout_seconds,
                    )
                )
        text = "\n".join(page for page in pages if page).strip()
    except Exception:
        return LocalImageOcrResult(
            available=True,
            attempted=True,
            engine="tesseract_local",
            language=language,
            text_visibility="unavailable",
            error_bucket="local_ocr_failed",
        )
    return LocalImageOcrResult(
        available=True,
        attempted=True,
        text=text,
        engine="tesseract_local",
        language=language,
        text_visibility="recovered" if text else "not_recovered",
    )


def _choose_local_image_ocr_language() -> str:
    languages = _list_tesseract_languages()
    available = set(languages)
    if "rus" in available and "eng" in available:
        return "rus+eng"
    if "eng" in available:
        return "eng"
    if "rus" in available:
        return "rus"
    return "eng"


def _list_tesseract_languages() -> list[str]:
    binary = shutil_module.which("tesseract") or "tesseract"
    try:
        completed = subprocess.run(
            [binary, "--list-langs"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=10,
        )
    except Exception:
        return []
    if completed.returncode != 0:
        return []
    lines = (completed.stdout or completed.stderr or "").splitlines()
    return [line.strip() for line in lines if line.strip() and "available languages" not in line.lower()]


def _image_ocr_gate_marker(image_ocr: LocalImageOcrResult) -> dict[str, Any]:
    return {
        "ocr_gate_fallback_executed": bool(image_ocr.attempted and image_ocr.available),
        "ocr_gate_fallback_attempted": bool(image_ocr.attempted),
        "ocr_gate_fallback_engine": image_ocr.engine,
        "ocr_gate_fallback_language": image_ocr.language,
        "ocr_gate_fallback_text_visibility": image_ocr.text_visibility,
        "ocr_gate_fallback_review_only": True,
        "ocr_gate_fallback_auto_accept_allowed": False,
        "ocr_gate_fallback_error_bucket": image_ocr.error_bucket,
    }


def _review_bound_unavailable_file_result(
    source_path: Path,
    *,
    selected_extractor: str,
    error: str,
    image_ocr: LocalImageOcrResult | None = None,
    specialty: str = "",
    document_category: str = "",
) -> TestFileResult:
    destination = _move_to_unique_destination(source_path, TEST_REVIEW_DIR)
    image_ocr = image_ocr or LocalImageOcrResult(
        available=False,
        attempted=False,
        engine="tesseract_local",
        text_visibility="unavailable",
        error_bucket="local_ocr_unavailable",
    )
    return TestFileResult(
        file_name=source_path.name,
        status="review",
        outcome="queued_for_review",
        processed_path=str(destination),
        selected_extractor=selected_extractor,
        confidence=None,
        validation_status="extractor_unavailable",
        document_type=UNKNOWN_DOCUMENT_LABEL,
        operator_review_reason="manual_review_required",
        operator_reason_label="Manual review required",
        ocr_gate_review_only=True,
        ocr_gate_auto_accept_allowed=False,
        ocr_gate_fallback_review_only=True,
        ocr_gate_fallback_auto_accept_allowed=False,
        ocr_gate_fallback_executed=False,
        ocr_gate_fallback_engine=image_ocr.engine,
        ocr_gate_fallback_language=image_ocr.language,
        ocr_gate_fallback_text_visibility=image_ocr.text_visibility,
        ocr_gate_fallback_error_bucket=image_ocr.error_bucket,
        image_ocr_available=image_ocr.available,
        image_ocr_attempted=image_ocr.attempted,
        image_ocr_engine=image_ocr.engine,
        image_ocr_text_visibility=image_ocr.text_visibility,
        image_ocr_review_only=True,
        image_ocr_auto_accept_allowed=False,
        files_processed=1,
        ocr_attempted=bool(image_ocr.attempted),
        ocr_available=bool(image_ocr.available),
        ocr_recovered=False,
        ocr_text_character_bucket="0",
        ocr_line_count_bucket="0",
        source_modality="image_ocr",
        selected_document_category=document_category,
        selected_specialty_domain=specialty,
        document_type_before_extraction=UNKNOWN_DOCUMENT_LABEL,
        document_type_after_extraction=UNKNOWN_DOCUMENT_LABEL,
        runtime_diagnostic_summary=_runtime_diagnostic_summary(
            ocr_recovered=False,
            extractor_dispatch_attempted=False,
            candidates_created=0,
            records_written=0,
            records_deduped=0,
            review_bound_records_written=0,
        ),
        external_api_used=False,
        error=error,
    )


def _review_bound_no_text_file_result(
    source_path: Path,
    *,
    image_ocr: LocalImageOcrResult,
    specialty: str = "",
    document_category: str = "",
) -> TestFileResult:
    destination = _move_to_unique_destination(source_path, TEST_REVIEW_DIR)
    return TestFileResult(
        file_name=source_path.name,
        status="review_ocr_quality",
        outcome="queued_for_review",
        processed_path=str(destination),
        selected_extractor="local_image_ocr:tesseract_local",
        validation_status="empty",
        document_type=UNKNOWN_DOCUMENT_LABEL,
        ocr_quality_band="no_text_found",
        language_text_visibility="not_recovered",
        ocr_gate_review_only=True,
        ocr_gate_auto_accept_allowed=False,
        ocr_gate_fallback_executed=True,
        ocr_gate_fallback_engine=image_ocr.engine,
        ocr_gate_fallback_language=image_ocr.language,
        ocr_gate_fallback_text_visibility=image_ocr.text_visibility,
        ocr_gate_fallback_review_only=True,
        ocr_gate_fallback_auto_accept_allowed=False,
        image_ocr_available=True,
        image_ocr_attempted=True,
        image_ocr_engine=image_ocr.engine,
        image_ocr_text_visibility=image_ocr.text_visibility,
        image_ocr_review_only=True,
        image_ocr_auto_accept_allowed=False,
        files_processed=1,
        ocr_attempted=True,
        ocr_available=True,
        ocr_recovered=False,
        ocr_text_character_bucket="0",
        ocr_line_count_bucket="0",
        source_modality="image_ocr",
        selected_document_category=document_category,
        selected_specialty_domain=specialty,
        document_type_before_extraction=UNKNOWN_DOCUMENT_LABEL,
        document_type_after_extraction=UNKNOWN_DOCUMENT_LABEL,
        runtime_diagnostic_summary=_runtime_diagnostic_summary(
            ocr_recovered=False,
            extractor_dispatch_attempted=False,
            candidates_created=0,
            records_written=0,
            records_deduped=0,
            review_bound_records_written=0,
        ),
        external_api_used=False,
        operator_review_reason="manual_review_required",
        operator_reason_label="Manual review required",
        error="Local image OCR found no readable text.",
    )


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fallback_diagnostic_document_type(ocr_gate_marker: dict[str, Any]) -> str | None:
    family_diagnostic = ocr_gate_marker.get("document_family_classification_diagnostic")
    if isinstance(family_diagnostic, dict):
        family_candidate = family_diagnostic.get("candidate_family")
        if family_candidate and str(family_candidate).strip().lower() != "unknown":
            return str(family_candidate)
    for key in (
        "ocr_gate_fallback_treatment_classification_diagnostic",
        "ocr_gate_fallback_classification_diagnostic",
    ):
        diagnostic = ocr_gate_marker.get(key)
        if not isinstance(diagnostic, dict):
            continue
        candidate = diagnostic.get("matched_document_type_candidate")
        if candidate and str(candidate).strip().lower() != "unknown":
            return str(candidate)
    return None


def _runtime_family_diagnostic(extractor_result: dict[str, Any]) -> dict[str, Any] | None:
    direct = extractor_result.get("document_family_classification_diagnostic")
    if isinstance(direct, dict):
        return direct
    fallback = extractor_result.get("ocr_gate_fallback_classification_diagnostic")
    if isinstance(fallback, dict):
        nested = fallback.get("document_family_classification_diagnostic")
        if isinstance(nested, dict):
            return nested
    return None


def _runtime_document_type_candidate(
    audit: dict[str, Any],
    extractor_result: dict[str, Any],
    ocr_gate_marker: dict[str, Any],
) -> str | None:
    for value in (
        audit.get("document_type"),
        extractor_result.get("document_type"),
    ):
        if value and display_document_type(value) != UNKNOWN_DOCUMENT_LABEL:
            return str(value)
    return _fallback_diagnostic_document_type(ocr_gate_marker)


def _document_type_from_extracted_candidates(extractor_result: dict[str, Any]) -> str | None:
    preview = extractor_result.get("extracted_medical_facts_preview_safe")
    if not isinstance(preview, list) or not preview:
        return None
    kinds = {
        str(item.get("candidate_kind") or "").lower()
        for item in preview
        if isinstance(item, dict)
    }
    headings = {
        str(item.get("section_heading") or "").strip().lower()
        for item in preview
        if isinstance(item, dict)
    }
    if kinds & {"table_observation", "portal_card_result"}:
        return "Lab result"
    if headings & {"cytology", "pathology", "microscopic description", "gross description"}:
        return "Pathology report"
    if headings & {"findings", "impression", "procedure"}:
        return "Imaging report"
    if headings & {"plan", "recommendation", "recommendations", "treatment plan"}:
        return "Treatment plan"
    if kinds & {"key_value_field", "narrative_source_section"}:
        return "Clinical note"
    return None


def _ocr_text_shape_diagnostics(text: str) -> dict[str, Any]:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    return {
        "ocr_text_character_bucket": _count_bucket(len(str(text or "")), [(0, "0"), (500, "1-500"), (2000, "501-2000"), (5000, "2001-5000")], "5001+"),
        "ocr_line_count_bucket": _count_bucket(len(lines), [(0, "0"), (10, "1-10"), (50, "11-50"), (200, "51-200")], "201+"),
        "ocr_has_table_like_layout": any("|" in line or re.search(r"\S\s{2,}\S\s{2,}\S", line) for line in lines),
        "ocr_has_key_value_like_layout": any(re.search(r"^[^:\n]{1,60}:\s*\S", line) for line in lines),
        "ocr_has_section_heading_like_layout": any(
            re.search(r"^[A-Za-z][A-Za-z /-]{2,48}:$", line)
            or line.lower().rstrip(":") in {"findings", "impression", "results", "plan", "recommendation", "recommendations"}
            for line in lines
        ),
    }


def _count_bucket(value: int, bounds: list[tuple[int, str]], overflow: str) -> str:
    if value <= 0:
        return "0"
    previous = 0
    for upper, label in bounds:
        if upper == 0:
            continue
        if previous < value <= upper:
            return label
        previous = upper
    return overflow


def _drop_reason_counts(
    *,
    candidates_created: int,
    candidates_after_filter: int,
    records_deduped: int,
    ocr_recovered: bool,
    extractor_dispatch_attempted: bool,
) -> dict[str, int]:
    dropped = max(0, int(candidates_created or 0) - int(candidates_after_filter or 0))
    reasons: dict[str, int] = {}
    if records_deduped > 0:
        reasons["deduped_or_already_represented"] = int(records_deduped)
    remaining = max(0, dropped - int(records_deduped or 0))
    if remaining > 0:
        reasons["filtered_before_write"] = remaining
    if ocr_recovered and extractor_dispatch_attempted and int(candidates_created or 0) == 0:
        reasons["no_parseable_visible_observations"] = 1
    return reasons


def _runtime_diagnostic_summary(
    *,
    ocr_recovered: bool,
    extractor_dispatch_attempted: bool,
    candidates_created: int,
    records_written: int,
    records_deduped: int,
    review_bound_records_written: int,
) -> str:
    return (
        f"OCR recovered: {'yes' if ocr_recovered else 'no'}; "
        f"extractor dispatch: {'yes' if extractor_dispatch_attempted else 'no'}; "
        f"candidates: {int(candidates_created or 0)}; "
        f"written: {int(records_written or 0)}; "
        f"deduped: {int(records_deduped or 0)}; "
        f"review-bound written: {int(review_bound_records_written or 0)}"
    )


def _run_review_summary_from_extraction(extractor_result: dict[str, Any]) -> str:
    candidates = int(extractor_result.get("cross_domain_extraction_candidates_count") or 0)
    review_records = int(extractor_result.get("cross_domain_review_bound_records_written_count") or 0)
    deduped = int(extractor_result.get("cross_domain_records_deduped_count") or 0)
    if review_records > 0:
        return f"OCR recovered and structured review-bound observations created: {review_records}."
    if candidates > 0 and deduped > 0:
        return f"OCR recovered; {deduped} equivalent extraction candidate(s) were already represented."
    if candidates > 0:
        return "OCR recovered; visible extraction candidates require review."
    return "OCR recovered but no recognizable structured facts were created."


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
