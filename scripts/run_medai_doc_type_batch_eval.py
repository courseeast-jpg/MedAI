from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.document_type_registry import SUPPORTED_DOCUMENT_FAMILIES, UNKNOWN_DOCUMENT_LABEL
from app.lab_document_metadata import display_document_type, normalize_text_quality_label
from app.test_launcher import runtime_cyrillic_ocr_marker_for_result
from execution.audit import StageAuditLogger
from execution.logging import AuditLogger
from execution.pipeline import ExecutionPipeline


DEFAULT_OUTPUT_DIR = ROOT / "reports" / "medai_doc_type_eval_01"
REPORT_JSON_NAME = "medai_doc_type_eval_01_report.json"
REPORT_MD_NAME = "medai_doc_type_eval_01_report.md"
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}
REVIEW_BOUND_STATUSES = {"review", "review_ocr_quality", "no_text_found", "error"}
DEFAULT_TYPE_COUNTS = (
    "Lab result",
    "Imaging report",
    "Treatment plan",
    "Medication plan",
    "Clinical note",
    "Discharge summary",
    "Unknown",
)


@dataclass(frozen=True)
class EvaluationOptions:
    input_dir: Path
    output_dir: Path = DEFAULT_OUTPUT_DIR
    limit: int | None = None
    recursive: bool = False
    local_only: bool = True


def enumerate_supported_documents(input_dir: Path, *, recursive: bool = False, limit: int | None = None) -> list[Path]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError("Input directory does not exist or is not a directory.")
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    files = sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if limit is not None:
        return files[: max(0, int(limit))]
    return files


def run_batch_evaluation(
    options: EvaluationOptions,
    *,
    processor: Callable[[Path, str], Any] | None = None,
) -> dict[str, Any]:
    paths = enumerate_supported_documents(options.input_dir, recursive=options.recursive, limit=options.limit)
    if processor is None:
        processor = _build_pipeline_processor(local_only=options.local_only)

    records: list[dict[str, Any]] = []
    for index, path in enumerate(paths, start=1):
        safe_id = f"file_{index:03d}"
        try:
            result = processor(path, safe_id)
            records.append(build_safe_file_record(path, safe_id=safe_id, result=result))
        except Exception as exc:
            records.append(build_error_record(path, safe_id=safe_id, error=exc))

    report = build_report(records)
    write_reports(report, options.output_dir)
    return report


def _build_pipeline_processor(*, local_only: bool) -> Callable[[Path, str], Any]:
    if not local_only:
        raise ValueError("Only local-only evaluation is supported.")

    temp_dir = tempfile.TemporaryDirectory(prefix="medai_doc_type_eval_")
    temp_root = Path(temp_dir.name)
    pipeline = ExecutionPipeline(
        audit_logger=AuditLogger(temp_root / "execution_audit.jsonl"),
        stage_audit_logger=StageAuditLogger(temp_root / "pipeline_stages.jsonl"),
        review_queue_path=temp_root / "review_queue.jsonl",
    )
    setattr(pipeline, "_medai_doc_type_eval_temp_dir", temp_dir)

    def process(path: Path, safe_id: str) -> Any:
        session_id = f"doc-type-eval-{safe_id}-{uuid4()}"
        if path.suffix.lower() == ".pdf":
            return pipeline.process_pdf(path, specialty="general", session_id=session_id)
        if path.suffix.lower() == ".txt":
            text = path.read_text(encoding="utf-8", errors="replace")
            return pipeline.process_text(text, specialty="general", source_name=safe_id, session_id=session_id)
        raise ValueError("Unsupported document extension.")

    return process


def build_safe_file_record(path: Path, *, safe_id: str, result: Any) -> dict[str, Any]:
    extractor_payload = result.get("extractor_result", {}) if isinstance(result, dict) else getattr(result, "extractor_result", {})
    audit_payload = result.get("audit", {}) if isinstance(result, dict) else getattr(result, "audit", {})
    extractor_result = dict(extractor_payload or {})
    audit = dict(audit_payload or {})
    outcome = getattr(result, "outcome", None) or (result.get("outcome") if isinstance(result, dict) else None)
    validation_status = getattr(result, "validation_status", None) or (
        result.get("validation_status") if isinstance(result, dict) else None
    )
    ocr_marker = runtime_cyrillic_ocr_marker_for_result(extractor_result)
    family_diagnostic = _safe_family_diagnostic(extractor_result, ocr_marker)
    predicted_type = _predicted_document_type(audit, extractor_result, family_diagnostic)
    auto_accept_allowed = _any_auto_accept_allowed(extractor_result, ocr_marker, family_diagnostic)
    external_api_used = bool(
        extractor_result.get("external_api_used")
        or audit.get("external_api_used")
        or extractor_result.get("cloud_api_used")
        or audit.get("cloud_api_used")
    )

    return {
        "file_id": safe_id,
        "extension": path.suffix.lower(),
        "size_bucket": size_bucket(_safe_size(path)),
        "page_count_bucket": page_count_bucket(path),
        "ocr_quality_band": normalize_text_quality_label(
            audit.get("ocr_quality_band"),
            audit.get("input_quality_band"),
            extractor_result.get("ocr_quality_band"),
            extractor_result.get("input_quality_band"),
            extractor_result.get("text_quality_status"),
            audit.get("text_quality_status"),
        ),
        "language_text_visibility": ocr_marker.get("language_text_visibility"),
        "cyrillic_ocr_recommended": bool(ocr_marker.get("cyrillic_ocr_recommended", False)),
        "ocr_gate_fallback_executed": bool(ocr_marker.get("ocr_gate_fallback_executed", False)),
        "ocr_gate_fallback_cyrillic_detected": bool(
            ocr_marker.get("ocr_gate_fallback_cyrillic_detected", False)
        ),
        "ocr_gate_fallback_text_visibility": ocr_marker.get("ocr_gate_fallback_text_visibility"),
        "predicted_document_type": predicted_type,
        "document_family_candidate": _family_candidate(family_diagnostic, predicted_type),
        "matched_safe_cue_keys": _safe_string_list(family_diagnostic.get("matched_family_cue_keys")),
        "ambiguous_candidates": _safe_string_list(family_diagnostic.get("ambiguous_candidates")),
        "classification_block_reason": _safe_optional_string(
            family_diagnostic.get("classification_block_reason")
        ),
        "conflict_resolution_reason": _safe_optional_string(
            family_diagnostic.get("conflict_resolution_reason")
        ),
        "review_status": review_status(outcome, validation_status),
        "auto_accept_allowed": auto_accept_allowed,
        "external_api_used": external_api_used,
    }


def build_error_record(path: Path, *, safe_id: str, error: Exception) -> dict[str, Any]:
    return {
        "file_id": safe_id,
        "extension": path.suffix.lower(),
        "size_bucket": size_bucket(_safe_size(path)),
        "page_count_bucket": page_count_bucket(path),
        "ocr_quality_band": "unknown",
        "language_text_visibility": "unknown",
        "cyrillic_ocr_recommended": False,
        "ocr_gate_fallback_executed": False,
        "ocr_gate_fallback_cyrillic_detected": False,
        "ocr_gate_fallback_text_visibility": None,
        "predicted_document_type": UNKNOWN_DOCUMENT_LABEL,
        "document_family_candidate": UNKNOWN_DOCUMENT_LABEL,
        "matched_safe_cue_keys": [],
        "ambiguous_candidates": [],
        "classification_block_reason": "runtime_error",
        "conflict_resolution_reason": "none",
        "review_status": "error",
        "auto_accept_allowed": False,
        "external_api_used": False,
        "error_bucket": safe_error_bucket(error),
    }


def build_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    type_counts = {label: 0 for label in DEFAULT_TYPE_COUNTS}
    for family in SUPPORTED_DOCUMENT_FAMILIES:
        type_counts.setdefault(family, 0)
    for record in records:
        label = str(record.get("predicted_document_type") or UNKNOWN_DOCUMENT_LABEL)
        type_counts[label] = type_counts.get(label, 0) + 1

    review_counts = Counter(str(record.get("review_status") or "review") for record in records)
    cue_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        family = str(record.get("document_family_candidate") or UNKNOWN_DOCUMENT_LABEL)
        for cue in _safe_string_list(record.get("matched_safe_cue_keys")):
            cue_counts[family][cue] += 1

    top_cues = {
        family: [{"cue_key": cue, "count": count} for cue, count in counts.most_common(10)]
        for family, counts in sorted(cue_counts.items())
    }
    ambiguous_count = sum(1 for record in records if record.get("ambiguous_candidates"))
    fallback_count = sum(1 for record in records if record.get("ocr_gate_fallback_executed") is True)
    external_api_count = sum(1 for record in records if record.get("external_api_used") is True)
    auto_accept_count = sum(1 for record in records if record.get("auto_accept_allowed") is True)

    return {
        "conclusion": "medai_doc_type_eval_01_ready",
        "generated_at": datetime.now(UTC).isoformat(),
        "evaluation_type": "privacy_safe_local_batch_document_type_evaluation",
        "total_files_evaluated": len(records),
        "count_by_predicted_document_type": type_counts,
        "count_by_review_status": dict(sorted(review_counts.items())),
        "ambiguous_family_conflict_count": ambiguous_count,
        "ocr_fallback_used_count": fallback_count,
        "external_api_used_count": external_api_count,
        "auto_accept_allowed_count": auto_accept_count,
        "top_safe_cue_categories_by_family": top_cues,
        "anonymous_per_file_table": records,
        "recommended_next_actions": recommended_next_actions(records),
        "runtime_behavior_changed": False,
        "ocr_routing_changed": False,
        "ocr_engine_changed": False,
        "document_family_classifier_logic_changed": False,
        "cue_thresholds_changed": False,
        "conflict_rules_changed": False,
        "auto_acceptance_changed": False,
        "clinical_interpretation_added": False,
        "medication_parsing_added": False,
        "dose_parsing_added": False,
        "lab_value_parsing_added": False,
        "imaging_interpretation_added": False,
        "ddi_logic_changed": False,
        "external_api_enabled": False,
        "raw_ocr_text_in_public_reports": False,
        "raw_document_text_in_public_reports": False,
        "raw_filenames_private_paths_in_public_reports": False,
        "private_files_staged": False,
        "source_documents_staged": False,
        "runtime_db_staged": False,
    }


def recommended_next_actions(records: list[dict[str, Any]]) -> dict[str, int]:
    actions = {
        "cue-pack update needed": 0,
        "conflict-resolution update needed": 0,
        "UI-only issue": 0,
        "leave Unknown/manual review": 0,
    }
    for record in records:
        predicted = str(record.get("predicted_document_type") or UNKNOWN_DOCUMENT_LABEL)
        block_reason = str(record.get("classification_block_reason") or "")
        if record.get("ambiguous_candidates"):
            actions["conflict-resolution update needed"] += 1
        elif predicted == UNKNOWN_DOCUMENT_LABEL and block_reason in {
            "too_few_safe_family_cue_keys",
            "too_few_safe_lab_cue_keys",
            "too_few_safe_treatment_cue_keys",
        }:
            actions["cue-pack update needed"] += 1
        elif predicted == UNKNOWN_DOCUMENT_LABEL:
            actions["leave Unknown/manual review"] += 1
        else:
            actions["UI-only issue"] += 0
    return actions


def write_reports(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / REPORT_JSON_NAME
    md_path = output_dir / REPORT_MD_NAME
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown_report(report), encoding="utf-8")
    return json_path, md_path


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# MEDAI-DOC-TYPE-EVAL-01",
        "",
        f"- Conclusion: `{report['conclusion']}`",
        f"- Total files evaluated: `{report['total_files_evaluated']}`",
        f"- External API used count: `{report['external_api_used_count']}`",
        f"- Auto-accept allowed count: `{report['auto_accept_allowed_count']}`",
        f"- OCR fallback used count: `{report['ocr_fallback_used_count']}`",
        f"- Ambiguous family conflict count: `{report['ambiguous_family_conflict_count']}`",
        "",
        "## Count By Predicted Document Type",
        "",
    ]
    for label, count in report["count_by_predicted_document_type"].items():
        lines.append(f"- {label}: `{count}`")
    lines.extend(["", "## Count By Review Status", ""])
    for label, count in report["count_by_review_status"].items():
        lines.append(f"- {label}: `{count}`")
    lines.extend(["", "## Recommended Next Actions", ""])
    for label, count in report["recommended_next_actions"].items():
        lines.append(f"- {label}: `{count}`")
    lines.extend(["", "## Anonymous Per-File Table", ""])
    if not report["anonymous_per_file_table"]:
        lines.append("- No supported files evaluated.")
    for record in report["anonymous_per_file_table"]:
        lines.append(
            "- "
            f"{record['file_id']} "
            f"extension=`{record['extension']}` "
            f"type=`{record['predicted_document_type']}` "
            f"review_status=`{record['review_status']}` "
            f"fallback=`{record['ocr_gate_fallback_executed']}` "
            f"external_api=`{record['external_api_used']}`"
        )
    lines.extend(
        [
            "",
            "## Safety / Privacy",
            "",
            "- No raw OCR text.",
            "- No raw document text.",
            "- No raw filenames.",
            "- No private paths.",
            "- No PHI.",
            "- No secrets.",
            "- Source documents staged: false.",
            "- External APIs enabled: false.",
            "",
        ]
    )
    return "\n".join(lines)


def _predicted_document_type(
    audit: dict[str, Any],
    extractor_result: dict[str, Any],
    family_diagnostic: dict[str, Any],
) -> str:
    for value in (
        audit.get("document_type"),
        extractor_result.get("document_type"),
        family_diagnostic.get("candidate_family"),
    ):
        label = display_document_type(value)
        if label != UNKNOWN_DOCUMENT_LABEL:
            return label
    return UNKNOWN_DOCUMENT_LABEL


def _safe_family_diagnostic(extractor_result: dict[str, Any], ocr_marker: dict[str, Any]) -> dict[str, Any]:
    direct = extractor_result.get("document_family_classification_diagnostic")
    if isinstance(direct, dict):
        return _sanitize_family_diagnostic(direct)
    marker_direct = ocr_marker.get("document_family_classification_diagnostic")
    if isinstance(marker_direct, dict):
        return _sanitize_family_diagnostic(marker_direct)
    fallback = extractor_result.get("ocr_gate_fallback_classification_diagnostic")
    if isinstance(fallback, dict):
        nested = fallback.get("document_family_classification_diagnostic")
        if isinstance(nested, dict):
            return _sanitize_family_diagnostic(nested)
    return _sanitize_family_diagnostic({})


def _sanitize_family_diagnostic(diagnostic: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_family": _safe_optional_string(diagnostic.get("candidate_family")) or UNKNOWN_DOCUMENT_LABEL,
        "matched_family_cue_keys": _safe_string_list(diagnostic.get("matched_family_cue_keys")),
        "matched_language_cue_groups": _safe_string_list(diagnostic.get("matched_language_cue_groups")),
        "ambiguous_candidates": _safe_string_list(diagnostic.get("ambiguous_candidates")),
        "classification_block_reason": _safe_optional_string(
            diagnostic.get("classification_block_reason")
        ) or "unknown",
        "conflict_resolution_reason": _safe_optional_string(
            diagnostic.get("conflict_resolution_reason")
        ) or "none",
        "review_only": True,
        "auto_accept_allowed": False,
    }


def _family_candidate(diagnostic: dict[str, Any], fallback: str) -> str:
    candidate = _safe_optional_string(diagnostic.get("candidate_family"))
    return candidate or fallback or UNKNOWN_DOCUMENT_LABEL


def _any_auto_accept_allowed(
    extractor_result: dict[str, Any],
    ocr_marker: dict[str, Any],
    family_diagnostic: dict[str, Any],
) -> bool:
    return bool(
        extractor_result.get("auto_accept_allowed")
        or extractor_result.get("ocr_gate_auto_accept_allowed")
        or extractor_result.get("ocr_gate_fallback_auto_accept_allowed")
        or ocr_marker.get("ocr_gate_auto_accept_allowed")
        or ocr_marker.get("ocr_gate_fallback_auto_accept_allowed")
        or family_diagnostic.get("auto_accept_allowed")
    )


def review_status(outcome: Any, validation_status: Any) -> str:
    outcome_value = str(outcome or "").lower()
    validation_value = str(validation_status or "").lower()
    if outcome_value == "written" or validation_value == "accepted":
        return "accepted"
    if "ocr" in outcome_value:
        return "review_ocr_quality"
    if outcome_value in {"error", "failed"}:
        return "error"
    if validation_value in {"rejected", "needs_review"} or outcome_value in {"queued_for_review", "blocked_ddi"}:
        return "review"
    return "review"


def size_bucket(size: int) -> str:
    if size <= 0:
        return "empty"
    if size < 100_000:
        return "small"
    if size < 2_000_000:
        return "medium"
    if size < 20_000_000:
        return "large"
    return "very_large"


def page_count_bucket(path: Path) -> str:
    if path.suffix.lower() != ".pdf":
        return "not_applicable"
    try:
        from PyPDF2 import PdfReader

        count = len(PdfReader(str(path)).pages)
    except Exception:
        return "unknown"
    if count <= 0:
        return "none"
    if count == 1:
        return "single"
    if count <= 5:
        return "few"
    return "many"


def safe_error_bucket(error: Exception) -> str:
    name = error.__class__.__name__.lower()
    if "permission" in name:
        return "permission_error"
    if "filenotfound" in name or "notfound" in name:
        return "file_not_found"
    if "value" in name:
        return "unsupported_or_invalid_input"
    return "processing_error"


def _safe_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0


def _safe_string_list(value: Any) -> list[str]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, dict)):
        return []
    return sorted(str(item) for item in value if _safe_optional_string(item))


def _safe_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def parse_args() -> EvaluationOptions:
    parser = argparse.ArgumentParser(description="Run a privacy-safe local document-type batch evaluation.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--local-only", default="true")
    args = parser.parse_args()
    local_only = str(args.local_only).strip().lower() not in {"false", "0", "no"}
    return EvaluationOptions(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        recursive=args.recursive,
        local_only=local_only,
    )


def main() -> None:
    options = parse_args()
    report = run_batch_evaluation(options)
    print(
        json.dumps(
            {
                "conclusion": report["conclusion"],
                "total_files_evaluated": report["total_files_evaluated"],
                "external_api_used_count": report["external_api_used_count"],
                "auto_accept_allowed_count": report["auto_accept_allowed_count"],
                "report_json": str(options.output_dir / REPORT_JSON_NAME),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
