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
from ingestion.cyrillic_ocr_gate import bucket_text_length


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
    raw_status = review_status(outcome, validation_status)
    accepted_source = accepted_status_source(outcome, validation_status, audit, extractor_result)
    internal_text = _internal_text_for_bucketing(extractor_result)
    native_length_bucket = native_text_length_bucket(extractor_result, audit, internal_text)
    pdf_text_layer = pdf_text_layer_detected(path)
    image_like = image_like_pdf(path, pdf_text_layer, native_length_bucket)
    raw_language_visibility = language_visibility_status(ocr_marker, extractor_result)
    cyrillic_visibility = cyrillic_visibility_status(ocr_marker, extractor_result)
    language_audit = language_visibility_audit(
        extractor_result=extractor_result,
        audit=audit,
        internal_text=internal_text,
        native_text_length_bucket=native_length_bucket,
        language_visibility_status=raw_language_visibility,
    )
    script_level_visibility = script_level_visibility_status(
        raw_language_visibility,
        language_audit["script_detection_result"],
    )
    language_visibility = script_level_visibility or raw_language_visibility
    fallback_executed = bool(ocr_marker.get("ocr_gate_fallback_executed", False))
    fallback_eligible = ocr_fallback_eligible(
        cyrillic_ocr_recommended=bool(ocr_marker.get("cyrillic_ocr_recommended", False)),
        image_like_pdf=image_like,
        language_visibility_status=raw_language_visibility,
        native_text_length_bucket=native_length_bucket,
        extension=path.suffix.lower(),
    )
    fallback_not_triggered = ocr_fallback_not_triggered_reason(
        extension=path.suffix.lower(),
        ocr_fallback_executed=fallback_executed,
        ocr_fallback_eligible=fallback_eligible,
        pdf_text_layer_detected=pdf_text_layer,
        image_like_pdf=image_like,
        native_text_length_bucket=native_length_bucket,
        language_visibility_status=raw_language_visibility,
        error_bucket=None,
    )
    cue_count_bucket = document_family_cue_count_bucket(family_diagnostic)
    status_mapping = normalize_review_status_for_document_type(
        predicted_type=predicted_type,
        raw_review_status=raw_status,
        accepted_status_source=accepted_source,
    )

    record = {
        "file_id": safe_id,
        "extension": path.suffix.lower(),
        "size_bucket": size_bucket(_safe_size(path)),
        "page_count_bucket": page_count_bucket(path),
        "text_extraction_status_bucket": text_extraction_status_bucket(
            normalize_text_quality_label(
                audit.get("ocr_quality_band"),
                audit.get("input_quality_band"),
                extractor_result.get("ocr_quality_band"),
                extractor_result.get("input_quality_band"),
                extractor_result.get("text_quality_status"),
                audit.get("text_quality_status"),
            ),
            native_length_bucket,
            error_bucket=None,
        ),
        "native_text_length_bucket": native_length_bucket,
        "pdf_text_layer_detected": pdf_text_layer,
        "image_like_pdf": image_like,
        "ocr_fallback_eligible": fallback_eligible,
        "ocr_fallback_not_triggered_reason": fallback_not_triggered,
        "language_visibility_status": language_visibility,
        "raw_language_visibility_status": raw_language_visibility,
        "language_script_visibility": script_level_visibility or "not_applicable",
        "cyrillic_visibility_status": cyrillic_visibility,
        "document_family_cue_count_bucket": cue_count_bucket,
        "language_script_detector_unknown_bucket": language_script_detector_unknown_bucket(language_audit),
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
        "ocr_gate_fallback_executed": fallback_executed,
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
        "review_status": status_mapping["review_status"],
        "raw_review_status": raw_status,
        "accepted_status_source": accepted_source,
        "status_mapping_action": status_mapping["status_mapping_action"],
        "status_mapping_note": status_mapping["status_mapping_note"],
        "text_source_present": language_audit["text_source_present"],
        "text_extraction_attempted": language_audit["text_extraction_attempted"],
        "text_extraction_result_bucket": language_audit["text_extraction_result_bucket"],
        "language_detector_attempted": language_audit["language_detector_attempted"],
        "language_detector_input_bucket": language_audit["language_detector_input_bucket"],
        "script_detection_attempted": language_audit["script_detection_attempted"],
        "script_detection_result": language_audit["script_detection_result"],
        "alphabetic_content_bucket": language_audit["alphabetic_content_bucket"],
        "numeric_content_bucket": language_audit["numeric_content_bucket"],
        "symbol_content_bucket": language_audit["symbol_content_bucket"],
        "garbled_text_detected": language_audit["garbled_text_detected"],
        "detector_confidence_bucket": language_audit["detector_confidence_bucket"],
        "visibility_unknown_reason": language_audit["visibility_unknown_reason"],
        "auto_accept_allowed": auto_accept_allowed,
        "external_api_used": external_api_used,
    }
    record["unknown_failure_bucket"] = unknown_failure_bucket(record)
    record["unknown_ocr_routing_bucket"] = unknown_ocr_routing_bucket(record)
    record["status_consistency_flags"] = status_consistency_flags(record)
    return record


def build_error_record(path: Path, *, safe_id: str, error: Exception) -> dict[str, Any]:
    error_bucket = safe_error_bucket(error)
    return {
        "file_id": safe_id,
        "extension": path.suffix.lower(),
        "size_bucket": size_bucket(_safe_size(path)),
        "page_count_bucket": page_count_bucket(path),
        "text_extraction_status_bucket": "extraction_error",
        "native_text_length_bucket": "unknown",
        "pdf_text_layer_detected": pdf_text_layer_detected(path),
        "image_like_pdf": "unknown",
        "ocr_fallback_eligible": "unknown",
        "ocr_fallback_not_triggered_reason": "extraction_error",
        "language_visibility_status": "unknown",
        "raw_language_visibility_status": "unknown",
        "language_script_visibility": "not_applicable",
        "cyrillic_visibility_status": "unknown",
        "document_family_cue_count_bucket": "none",
        "language_script_detector_unknown_bucket": "detector_unknown_unclassified",
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
        "raw_review_status": "error",
        "accepted_status_source": "not_accepted",
        "status_mapping_action": "not_applicable",
        "status_mapping_note": "not_applicable",
        "text_source_present": "unknown",
        "text_extraction_attempted": "yes",
        "text_extraction_result_bucket": "unknown",
        "language_detector_attempted": "no",
        "language_detector_input_bucket": "unknown",
        "script_detection_attempted": "no",
        "script_detection_result": "unknown",
        "alphabetic_content_bucket": "unknown",
        "numeric_content_bucket": "unknown",
        "symbol_content_bucket": "unknown",
        "garbled_text_detected": "unknown",
        "detector_confidence_bucket": "unknown",
        "visibility_unknown_reason": "metadata_missing",
        "auto_accept_allowed": False,
        "external_api_used": False,
        "error_bucket": error_bucket,
        "unknown_failure_bucket": "unsupported_or_empty_text",
        "unknown_ocr_routing_bucket": "extraction_error",
        "status_consistency_flags": [],
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
    accepted_count = sum(1 for record in records if record.get("review_status") == "accepted")
    status_anomalies = status_anomaly_records(records)
    accepted_sources = Counter(str(record.get("accepted_status_source") or "not_accepted") for record in records)
    unknown_diagnostics = unknown_diagnostic_summary(records)
    evaluation_status = "failed" if external_api_count > 0 or auto_accept_count > 0 else "passed"

    return {
        "conclusion": "medai_doc_type_eval_01_ready",
        "evaluation_status": evaluation_status,
        "generated_at": datetime.now(UTC).isoformat(),
        "evaluation_type": "privacy_safe_local_batch_document_type_evaluation",
        "total_files_evaluated": len(records),
        "count_by_predicted_document_type": type_counts,
        "count_by_review_status": dict(sorted(review_counts.items())),
        "ambiguous_family_conflict_count": ambiguous_count,
        "ocr_fallback_used_count": fallback_count,
        "external_api_used_count": external_api_count,
        "auto_accept_allowed_count": auto_accept_count,
        "accepted_count": accepted_count,
        "accepted_status_source_counts": dict(sorted(accepted_sources.items())),
        "status_consistency": {
            "unknown_accepted_anomaly_count": len(status_anomalies),
            "unknown_accepted_anomaly_file_ids": [str(record["file_id"]) for record in status_anomalies],
            "invalid_status_mapping_file_ids": [
                str(record["file_id"])
                for record in records
                if "invalid_status_mapping_normalized" in _safe_string_list(
                    record.get("status_consistency_flags")
                )
            ],
            "policy_anomaly_present": bool(status_anomalies),
        },
        "unknown_diagnostics": unknown_diagnostics,
        "unknown_ocr_routing_diagnostics": unknown_ocr_routing_summary(records),
        "language_visibility_audit": language_visibility_audit_summary(records),
        "language_script_detector_unknown_diagnostics": language_script_detector_unknown_summary(records),
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
        "OCR/text visibility investigation": 0,
        "status mapping fix": 0,
        "external API violation": 0,
        "auto-accept violation": 0,
        "UI-only issue": 0,
        "leave Unknown/manual review": 0,
    }
    unknown_summary = unknown_diagnostic_summary(records)
    if unknown_summary["unknown_accepted_status_anomaly_count"]:
        actions["status mapping fix"] += int(unknown_summary["unknown_accepted_status_anomaly_count"])
    if any(record.get("external_api_used") is True for record in records):
        actions["external API violation"] += 1
    if any(record.get("auto_accept_allowed") is True for record in records):
        actions["auto-accept violation"] += 1
    if (
        unknown_summary["unknown_with_low_or_incomplete_text_visibility"]
        > unknown_summary["unknown_with_high_text_visibility"]
    ):
        actions["OCR/text visibility investigation"] += 1
    routing_summary = unknown_ocr_routing_summary(records)
    if (
        routing_summary["unknown_image_like_pdfs_not_routed_to_ocr"]
        or routing_summary["unknown_files_eligible_for_fallback_but_not_triggered"]
        or routing_summary["unknown_files_with_extraction_errors"]
    ):
        actions["OCR/text visibility investigation"] += 1
    if unknown_summary["unknown_with_any_family_cue_keys"] > 0:
        actions["cue-pack update needed"] += int(unknown_summary["unknown_with_any_family_cue_keys"])
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
            if not record.get("matched_safe_cue_keys"):
                actions["leave Unknown/manual review"] += 1
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
        f"- Accepted count: `{report['accepted_count']}`",
        f"- Auto-accept allowed count: `{report['auto_accept_allowed_count']}`",
        f"- OCR fallback used count: `{report['ocr_fallback_used_count']}`",
        f"- Ambiguous family conflict count: `{report['ambiguous_family_conflict_count']}`",
        f"- Unknown accepted anomaly count: `{report['status_consistency']['unknown_accepted_anomaly_count']}`",
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
    lines.extend(["", "## Unknown Diagnostics", ""])
    unknown = report["unknown_diagnostics"]
    for key in (
        "unknown_with_fallback_true",
        "unknown_with_fallback_false",
        "unknown_with_high_text_visibility",
        "unknown_with_low_or_incomplete_text_visibility",
        "unknown_with_any_family_cue_keys",
        "unknown_with_no_family_cue_keys",
        "unknown_accepted_status_anomaly_count",
    ):
        lines.append(f"- {key}: `{unknown[key]}`")
    lines.extend(["", "### Unknown Failure Buckets", ""])
    for bucket, count in unknown["unknown_failure_bucket_counts"].items():
        lines.append(f"- {bucket}: `{count}`")
    lines.extend(["", "### Priority Unknown Samples", ""])
    if not unknown["priority_unknown_samples"]:
        lines.append("- No Unknown samples.")
    for item in unknown["priority_unknown_samples"]:
        lines.append(
            "- "
            f"{item['file_id']} "
            f"bucket=`{item['unknown_failure_bucket']}` "
            f"fallback=`{item['ocr_gate_fallback_executed']}` "
            f"visibility=`{item['language_text_visibility']}` "
            f"accepted_source=`{item['accepted_status_source']}`"
        )
    lines.extend(["", "### Unknown OCR Routing Diagnostics", ""])
    routing = report["unknown_ocr_routing_diagnostics"]
    for key in (
        "unknown_image_like_pdfs_not_routed_to_ocr",
        "unknown_text_layer_pdfs_with_too_little_text",
        "unknown_files_with_extraction_errors",
        "unknown_files_eligible_for_fallback_but_not_triggered",
    ):
        lines.append(f"- {key}: `{routing[key]}`")
    lines.extend(["", "#### Fallback False Unknown Buckets", ""])
    for bucket, count in routing["fallback_false_unknown_bucket_counts"].items():
        lines.append(f"- {bucket}: `{count}`")
    lines.extend(["", "#### Not Fallback Eligible Reasons", ""])
    for reason, count in routing["unknown_not_fallback_eligible_reason_counts"].items():
        lines.append(f"- {reason}: `{count}`")
    lines.extend(["", "### Language Visibility Audit", ""])
    language_audit = report["language_visibility_audit"]
    for key in (
        "unknown_visibility_no_text_available",
        "unknown_visibility_text_not_passed_to_detector",
        "unknown_visibility_detector_not_called",
        "unknown_visibility_numeric_or_symbol_only_text",
        "unknown_visibility_metadata_missing",
    ):
        lines.append(f"- {key}: `{language_audit[key]}`")
    lines.extend(["", "#### Visibility Unknown Reasons", ""])
    for reason, count in language_audit["visibility_unknown_reason_counts"].items():
        lines.append(f"- {reason}: `{count}`")
    lines.extend(["", "### Language / Script Detector Unknown Diagnostics", ""])
    detector = report["language_script_detector_unknown_diagnostics"]
    for key in (
        "detector_returned_unknown_total",
        "numeric_heavy",
        "alphabetic_low",
        "cyrillic_present_but_language_unknown",
        "latin_present_but_language_unknown",
        "mixed_script",
        "detector_output_missing",
        "detector_low_confidence",
    ):
        lines.append(f"- {key}: `{detector[key]}`")
    lines.extend(["", "#### Detector Unknown Buckets", ""])
    for bucket, count in detector["detector_unknown_bucket_counts"].items():
        lines.append(f"- {bucket}: `{count}`")
    lines.extend(["", "#### Top Detector Unknown Samples", ""])
    if not detector["top_detector_unknown_samples"]:
        lines.append("- No detector-unknown samples.")
    for item in detector["top_detector_unknown_samples"]:
        lines.append(
            "- "
            f"{item['file_id']} "
            f"bucket=`{item['language_script_detector_unknown_bucket']}` "
            f"script=`{item['script_detection_result']}` "
            f"script_visibility=`{item['language_script_visibility']}`"
        )
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
            f"unknown_bucket=`{record['unknown_failure_bucket']}` "
            f"ocr_route_bucket=`{record['unknown_ocr_routing_bucket']}` "
            f"visibility_reason=`{record['visibility_unknown_reason']}` "
            f"detector_bucket=`{record['language_script_detector_unknown_bucket']}` "
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


def accepted_status_source(
    outcome: Any,
    validation_status: Any,
    audit: dict[str, Any] | None = None,
    extractor_result: dict[str, Any] | None = None,
) -> str:
    outcome_value = str(outcome or "").lower()
    validation_value = str(validation_status or "").lower()
    audit = audit or {}
    extractor_result = extractor_result or {}
    prior_status = str(
        audit.get("prior_record_status")
        or extractor_result.get("prior_record_status")
        or audit.get("record_status")
        or extractor_result.get("record_status")
        or ""
    ).lower()
    if validation_value == "accepted":
        return "runtime_validation_status"
    if outcome_value == "written":
        return "runtime_outcome"
    if prior_status in {"accepted", "active", "written"}:
        return "prior_record_status"
    return "not_accepted"


def normalize_review_status_for_document_type(
    *,
    predicted_type: str,
    raw_review_status: str,
    accepted_status_source: str,
) -> dict[str, str]:
    if predicted_type == UNKNOWN_DOCUMENT_LABEL and raw_review_status == "accepted":
        if accepted_status_source == "prior_record_status":
            return {
                "review_status": "accepted",
                "status_mapping_action": "historical_prior_status_preserved",
                "status_mapping_note": "Unknown accepted status came from prior record status.",
            }
        return {
            "review_status": "review",
            "status_mapping_action": "normalized_unknown_runtime_accepted_to_review",
            "status_mapping_note": "Unknown runtime accepted status is invalid for batch evaluation and was reported as review.",
        }
    return {
        "review_status": raw_review_status,
        "status_mapping_action": "unchanged",
        "status_mapping_note": "not_applicable",
    }


def unknown_failure_bucket(record: dict[str, Any]) -> str:
    if str(record.get("predicted_document_type") or "") != UNKNOWN_DOCUMENT_LABEL:
        return "not_applicable"
    if record.get("status_mapping_action") == "normalized_unknown_runtime_accepted_to_review":
        return "status_mapping_anomaly"
    if record.get("review_status") == "accepted":
        return "status_mapping_anomaly"
    if str(record.get("size_bucket") or "") == "empty" or str(record.get("ocr_quality_band") or "") in {
        "No text found",
        "Failed",
    }:
        return "unsupported_or_empty_text"
    if record.get("ambiguous_candidates"):
        return "ambiguous_below_threshold"
    if record.get("ocr_gate_fallback_executed") is True:
        return "fallback_ran_but_no_family_match"
    if record.get("cyrillic_ocr_recommended") is True:
        return "OCR_not_triggered"
    visibility = str(record.get("language_text_visibility") or "").lower()
    if visibility in {"", "none", "unknown", "incomplete", "not_applicable", "failed"}:
        return "insufficient_text_visibility"
    if record.get("matched_safe_cue_keys"):
        return "generic_cues_only"
    return "no_safe_document_family_cues"


def unknown_ocr_routing_bucket(record: dict[str, Any]) -> str:
    if str(record.get("predicted_document_type") or "") != UNKNOWN_DOCUMENT_LABEL:
        return "not_applicable"
    if record.get("ocr_gate_fallback_executed") is True:
        return "fallback_executed"
    if record.get("review_status") == "accepted":
        return "status_mapping_anomaly"
    reason = str(record.get("ocr_fallback_not_triggered_reason") or "unknown_reason")
    allowed = {
        "no_text_layer",
        "text_layer_present_but_too_short",
        "image_like_pdf_but_not_routed_to_ocr",
        "language_visibility_unknown",
        "unsupported_pdf_structure",
        "extraction_error",
        "routing_not_eligible",
        "unknown_reason",
    }
    return reason if reason in allowed else "unknown_reason"


def status_consistency_flags(record: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if str(record.get("predicted_document_type") or "") == UNKNOWN_DOCUMENT_LABEL:
        if record.get("status_mapping_action") == "normalized_unknown_runtime_accepted_to_review":
            flags.append("invalid_status_mapping_normalized")
        elif record.get("review_status") == "accepted":
            flags.append("unknown_accepted_historical_prior_status")
    if record.get("external_api_used") is True:
        flags.append("external_api_used_policy_failure")
    if record.get("auto_accept_allowed") is True:
        flags.append("auto_accept_allowed_policy_failure")
    return flags


def status_anomaly_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if "invalid_status_mapping_normalized" in _safe_string_list(record.get("status_consistency_flags"))
    ]


def unknown_diagnostic_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    unknown_records = [
        record for record in records if str(record.get("predicted_document_type") or "") == UNKNOWN_DOCUMENT_LABEL
    ]
    bucket_counts = Counter(str(record.get("unknown_failure_bucket") or "unknown") for record in unknown_records)
    high_visibility_values = {"visible", "recovered", "readable"}
    high_visibility = [
        record
        for record in unknown_records
        if str(record.get("language_text_visibility") or "").lower() in high_visibility_values
    ]
    low_visibility = [record for record in unknown_records if record not in high_visibility]
    any_cues = [record for record in unknown_records if record.get("matched_safe_cue_keys")]
    no_cues = [record for record in unknown_records if not record.get("matched_safe_cue_keys")]
    samples = sorted(
        unknown_records,
        key=lambda record: (
            _unknown_priority_rank(str(record.get("unknown_failure_bucket") or "")),
            str(record.get("file_id") or ""),
        ),
    )[:10]
    return {
        "unknown_total": len(unknown_records),
        "unknown_with_fallback_true": sum(
            1 for record in unknown_records if record.get("ocr_gate_fallback_executed") is True
        ),
        "unknown_with_fallback_false": sum(
            1 for record in unknown_records if record.get("ocr_gate_fallback_executed") is not True
        ),
        "unknown_with_high_text_visibility": len(high_visibility),
        "unknown_with_low_or_incomplete_text_visibility": len(low_visibility),
        "unknown_with_any_family_cue_keys": len(any_cues),
        "unknown_with_no_family_cue_keys": len(no_cues),
        "unknown_accepted_status_anomaly_count": len(status_anomaly_records(unknown_records)),
        "unknown_failure_bucket_counts": dict(sorted(bucket_counts.items())),
        "priority_unknown_samples": [
            {
                "file_id": str(record.get("file_id")),
                "unknown_failure_bucket": str(record.get("unknown_failure_bucket")),
                "language_text_visibility": record.get("language_text_visibility"),
                "language_visibility_status": record.get("language_visibility_status"),
                "cyrillic_visibility_status": record.get("cyrillic_visibility_status"),
                "native_text_length_bucket": record.get("native_text_length_bucket"),
                "pdf_text_layer_detected": record.get("pdf_text_layer_detected"),
                "image_like_pdf": record.get("image_like_pdf"),
                "ocr_fallback_eligible": record.get("ocr_fallback_eligible"),
                "ocr_fallback_not_triggered_reason": record.get("ocr_fallback_not_triggered_reason"),
                "unknown_ocr_routing_bucket": record.get("unknown_ocr_routing_bucket"),
                "ocr_gate_fallback_executed": bool(record.get("ocr_gate_fallback_executed")),
                "matched_safe_cue_keys": _safe_string_list(record.get("matched_safe_cue_keys")),
                "ambiguous_candidates": _safe_string_list(record.get("ambiguous_candidates")),
                "accepted_status_source": str(record.get("accepted_status_source") or "not_accepted"),
                "status_mapping_action": str(record.get("status_mapping_action") or "unknown"),
                "visibility_unknown_reason": str(record.get("visibility_unknown_reason") or "unknown"),
            }
            for record in samples
        ],
    }


def _unknown_priority_rank(bucket: str) -> int:
    order = {
        "status_mapping_anomaly": 0,
        "fallback_ran_but_no_family_match": 1,
        "generic_cues_only": 2,
        "ambiguous_below_threshold": 3,
        "OCR_not_triggered": 4,
        "insufficient_text_visibility": 5,
        "no_safe_document_family_cues": 6,
        "unsupported_or_empty_text": 7,
    }
    return order.get(bucket, 99)


def unknown_ocr_routing_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    unknown_records = [
        record for record in records if str(record.get("predicted_document_type") or "") == UNKNOWN_DOCUMENT_LABEL
    ]
    fallback_false_unknown = [
        record for record in unknown_records if record.get("ocr_gate_fallback_executed") is not True
    ]
    not_eligible = [
        record
        for record in fallback_false_unknown
        if str(record.get("ocr_fallback_eligible") or "unknown") != "yes"
    ]
    return {
        "unknown_image_like_pdfs_not_routed_to_ocr": sum(
            1
            for record in fallback_false_unknown
            if record.get("image_like_pdf") == "yes"
            and record.get("ocr_gate_fallback_executed") is not True
        ),
        "unknown_text_layer_pdfs_with_too_little_text": sum(
            1
            for record in fallback_false_unknown
            if record.get("pdf_text_layer_detected") == "yes"
            and record.get("native_text_length_bucket") in {"none", "tiny", "short"}
        ),
        "unknown_files_with_extraction_errors": sum(
            1 for record in unknown_records if record.get("text_extraction_status_bucket") == "extraction_error"
        ),
        "unknown_files_eligible_for_fallback_but_not_triggered": sum(
            1
            for record in fallback_false_unknown
            if record.get("ocr_fallback_eligible") == "yes"
        ),
        "fallback_false_unknown_bucket_counts": dict(
            sorted(Counter(str(record.get("unknown_ocr_routing_bucket") or "unknown_reason") for record in fallback_false_unknown).items())
        ),
        "unknown_not_fallback_eligible_reason_counts": dict(
            sorted(
                Counter(
                    str(record.get("ocr_fallback_not_triggered_reason") or "unknown_reason")
                    for record in not_eligible
                ).items()
            )
        ),
    }


def language_visibility_audit_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    unknown_records = [
        record for record in records if str(record.get("predicted_document_type") or "") == UNKNOWN_DOCUMENT_LABEL
    ]
    visibility_unknown = [
        record
        for record in unknown_records
        if str(record.get("visibility_unknown_reason") or "not_applicable") != "not_applicable"
    ]
    reason_counts = Counter(str(record.get("visibility_unknown_reason") or "unknown") for record in visibility_unknown)
    return {
        "unknown_visibility_total": len(visibility_unknown),
        "unknown_visibility_no_text_available": reason_counts.get("no_text_available", 0),
        "unknown_visibility_text_not_passed_to_detector": reason_counts.get(
            "text_not_passed_to_visibility_detector", 0
        ),
        "unknown_visibility_detector_not_called": reason_counts.get("detector_not_called", 0),
        "unknown_visibility_detector_returned_unknown": reason_counts.get("detector_returned_unknown", 0),
        "unknown_visibility_numeric_or_symbol_only_text": reason_counts.get("numeric_or_symbol_only_text", 0),
        "unknown_visibility_metadata_missing": reason_counts.get("metadata_missing", 0),
        "visibility_unknown_reason_counts": dict(sorted(reason_counts.items())),
    }


def language_script_detector_unknown_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    unknown_detector_records = [
        record
        for record in records
        if str(record.get("visibility_unknown_reason") or "") == "detector_returned_unknown"
    ]
    bucket_counts = Counter(
        str(record.get("language_script_detector_unknown_bucket") or "detector_unknown_unclassified")
        for record in unknown_detector_records
    )
    samples = sorted(
        unknown_detector_records,
        key=lambda record: (
            _detector_unknown_priority_rank(str(record.get("language_script_detector_unknown_bucket") or "")),
            str(record.get("file_id") or ""),
        ),
    )[:10]
    return {
        "detector_returned_unknown_total": len(unknown_detector_records),
        "numeric_heavy": bucket_counts.get("detector_input_numeric_heavy", 0),
        "alphabetic_low": sum(
            1
            for record in unknown_detector_records
            if record.get("alphabetic_content_bucket") in {"none", "low"}
        ),
        "cyrillic_present_but_language_unknown": bucket_counts.get("script_detectable_language_unknown", 0)
        + sum(
            1
            for record in unknown_detector_records
            if record.get("script_detection_result") == "cyrillic"
            and record.get("language_script_detector_unknown_bucket") != "script_detectable_language_unknown"
        ),
        "latin_present_but_language_unknown": sum(
            1 for record in unknown_detector_records if record.get("script_detection_result") == "latin"
        ),
        "mixed_script": bucket_counts.get("detector_input_mixed_script", 0),
        "detector_output_missing": bucket_counts.get("detector_output_not_propagated", 0),
        "detector_low_confidence": bucket_counts.get("detector_confidence_below_threshold", 0),
        "detector_unknown_bucket_counts": dict(sorted(bucket_counts.items())),
        "top_detector_unknown_samples": [
            {
                "file_id": str(record.get("file_id")),
                "language_script_detector_unknown_bucket": str(
                    record.get("language_script_detector_unknown_bucket")
                    or "detector_unknown_unclassified"
                ),
                "script_detection_result": str(record.get("script_detection_result") or "unknown"),
                "language_script_visibility": str(record.get("language_script_visibility") or "not_applicable"),
                "alphabetic_content_bucket": str(record.get("alphabetic_content_bucket") or "unknown"),
                "numeric_content_bucket": str(record.get("numeric_content_bucket") or "unknown"),
                "detector_confidence_bucket": str(record.get("detector_confidence_bucket") or "unknown"),
            }
            for record in samples
        ],
    }


def _detector_unknown_priority_rank(bucket: str) -> int:
    order = {
        "detector_output_not_propagated": 0,
        "script_detectable_language_unknown": 1,
        "detector_input_mixed_script": 2,
        "detector_input_numeric_heavy": 3,
        "detector_input_garbled_or_mojibake": 4,
        "detector_confidence_below_threshold": 5,
        "detector_input_tiny": 6,
        "detector_input_empty": 7,
        "detector_unknown_unclassified": 8,
    }
    return order.get(bucket, 99)


def _internal_text_for_bucketing(extractor_result: dict[str, Any]) -> str:
    for key in ("raw_text", "text", "native_text", "extracted_text"):
        value = extractor_result.get(key)
        if value:
            return str(value)
    return ""


def _has_text_source_field(extractor_result: dict[str, Any]) -> str:
    for key in ("raw_text", "text", "native_text", "extracted_text"):
        if key in extractor_result:
            return "yes" if str(extractor_result.get(key) or "").strip() else "no"
    return "unknown"


def native_text_length_bucket(extractor_result: dict[str, Any], audit: dict[str, Any], internal_text: str) -> str:
    for key in ("text_length_bucket", "native_text_length_bucket"):
        value = _safe_optional_string(extractor_result.get(key) or audit.get(key))
        if value in {"none", "tiny", "short", "medium", "long"}:
            return value
    for key in ("raw_text_length", "text_length", "ocr_text_length", "native_text_length"):
        value = extractor_result.get(key) if key in extractor_result else audit.get(key)
        if isinstance(value, (int, float)):
            return bucket_text_length(max(0, int(value)))
    return bucket_text_length(len(internal_text.strip()))


def text_extraction_status_bucket(quality_label: str, native_length_bucket: str, *, error_bucket: str | None) -> str:
    if error_bucket:
        return "extraction_error"
    quality = str(quality_label or "").lower()
    if "no text" in quality:
        return "no_text_available"
    if native_length_bucket in {"none", "tiny"}:
        return "text_too_short"
    if quality in {"readable", "clear", "ocr fallback used"} or native_length_bucket in {"medium", "long"}:
        return "text_visible"
    if quality in {"unknown", "not checked", "not available"}:
        return "unknown"
    return "low_text_visibility"


def language_visibility_audit(
    *,
    extractor_result: dict[str, Any],
    audit: dict[str, Any],
    internal_text: str,
    native_text_length_bucket: str,
    language_visibility_status: str,
) -> dict[str, str]:
    text_source = _has_text_source_field(extractor_result)
    extraction_attempted = "yes" if extractor_result or audit else "no"
    detector_attempted = _detector_attempted(extractor_result)
    detector_input_bucket = language_detector_input_bucket(internal_text)
    script_attempted = "yes" if detector_attempted == "yes" or internal_text.strip() else "no"
    script_result = script_detection_result(internal_text)
    alphabetic_bucket = content_density_bucket(_alphabetic_count(internal_text), internal_text)
    numeric_bucket = content_density_bucket(_digit_count(internal_text), internal_text)
    symbol_bucket = content_density_bucket(_symbol_count(internal_text), internal_text)
    garbled = garbled_text_detected(internal_text)
    confidence_bucket = detector_confidence_bucket(extractor_result)
    reason = visibility_unknown_reason(
        language_visibility_status=language_visibility_status,
        text_source_present=text_source,
        text_extraction_attempted=extraction_attempted,
        text_extraction_result_bucket=native_text_length_bucket,
        language_detector_attempted=detector_attempted,
        language_detector_input_bucket=detector_input_bucket,
        script_detection_result=script_result,
    )
    return {
        "text_source_present": text_source,
        "text_extraction_attempted": extraction_attempted,
        "text_extraction_result_bucket": native_text_length_bucket or "unknown",
        "language_detector_attempted": detector_attempted,
        "language_detector_input_bucket": detector_input_bucket,
        "script_detection_attempted": script_attempted,
        "script_detection_result": script_result,
        "alphabetic_content_bucket": alphabetic_bucket,
        "numeric_content_bucket": numeric_bucket,
        "symbol_content_bucket": symbol_bucket,
        "garbled_text_detected": garbled,
        "detector_confidence_bucket": confidence_bucket,
        "visibility_unknown_reason": reason,
    }


def _detector_attempted(extractor_result: dict[str, Any]) -> str:
    if extractor_result.get("language_support") or extractor_result.get("language_text_visibility"):
        return "yes"
    if any(
        key in extractor_result
        for key in (
            "detected_language",
            "script_detected",
            "language_confidence",
            "language_route_note",
        )
    ):
        return "yes"
    return "no"


def language_detector_input_bucket(text: str) -> str:
    length_bucket = bucket_text_length(len(str(text or "").strip()))
    if length_bucket in {"none"}:
        return "empty"
    if length_bucket == "tiny":
        return "tiny"
    if length_bucket == "short":
        return "short"
    if length_bucket in {"medium", "long"}:
        return "sufficient"
    return "unknown"


def script_detection_result(text: str) -> str:
    raw = str(text or "")
    latin = sum(1 for char in raw if "a" <= char.lower() <= "z")
    cyrillic = sum(1 for char in raw if "\u0400" <= char <= "\u04ff")
    digits = sum(1 for char in raw if char.isdigit())
    symbols = sum(1 for char in raw if not char.isalnum() and not char.isspace())
    if cyrillic and latin:
        return "mixed"
    if cyrillic:
        return "cyrillic"
    if latin:
        return "latin"
    if digits and not cyrillic and not latin:
        return "numeric_only"
    if symbols and not digits and not cyrillic and not latin:
        return "symbol_only"
    return "unknown"


def script_level_visibility_status(language_visibility_status: str, script_result: str) -> str | None:
    if language_visibility_status not in {"unknown", "incomplete", "not_applicable"}:
        return None
    if script_result == "cyrillic":
        return "cyrillic_visible_language_unknown"
    if script_result == "latin":
        return "latin_visible_language_unknown"
    if script_result == "mixed":
        return "mixed_script_visible_language_unknown"
    return None


def language_script_detector_unknown_bucket(audit: dict[str, str]) -> str:
    if audit.get("visibility_unknown_reason") == "not_applicable":
        return "not_applicable"
    if audit.get("text_source_present") == "unknown":
        return "detector_output_not_propagated"
    input_bucket = audit.get("language_detector_input_bucket")
    if input_bucket == "empty":
        return "detector_input_empty"
    if audit.get("numeric_content_bucket") == "high":
        return "detector_input_numeric_heavy"
    if audit.get("symbol_content_bucket") == "high":
        return "detector_input_symbol_heavy"
    if audit.get("script_detection_result") == "mixed":
        return "detector_input_mixed_script"
    if audit.get("garbled_text_detected") == "yes":
        return "detector_input_garbled_or_mojibake"
    if audit.get("detector_confidence_bucket") == "low":
        return "detector_confidence_below_threshold"
    if audit.get("script_detection_result") in {"cyrillic", "latin"}:
        return "script_detectable_language_unknown"
    if input_bucket == "tiny":
        return "detector_input_tiny"
    return "detector_unknown_unclassified"


def content_density_bucket(count: int, text: str) -> str:
    compact_len = max(1, len("".join(str(text or "").split())))
    ratio = count / compact_len
    if count <= 0:
        return "none"
    if ratio < 0.05:
        return "low"
    if ratio < 0.25:
        return "medium"
    return "high"


def _alphabetic_count(text: str) -> int:
    return sum(1 for char in str(text or "") if char.isalpha())


def _digit_count(text: str) -> int:
    return sum(1 for char in str(text or "") if char.isdigit())


def _symbol_count(text: str) -> int:
    return sum(1 for char in str(text or "") if not char.isalnum() and not char.isspace())


def garbled_text_detected(text: str) -> str:
    raw = str(text or "")
    if not raw:
        return "unknown"
    if "\ufffd" in raw or raw.count("?") >= 8 or raw.count("|") >= 8:
        return "yes"
    return "no"


def detector_confidence_bucket(extractor_result: dict[str, Any]) -> str:
    language_support = extractor_result.get("language_support")
    value = None
    if isinstance(language_support, dict):
        value = language_support.get("language_confidence")
    if value is None:
        value = extractor_result.get("language_confidence")
    if not isinstance(value, (int, float)):
        return "unknown"
    if float(value) < 0.5:
        return "low"
    if float(value) < 0.8:
        return "medium"
    return "high"


def visibility_unknown_reason(
    *,
    language_visibility_status: str,
    text_source_present: str,
    text_extraction_attempted: str,
    text_extraction_result_bucket: str,
    language_detector_attempted: str,
    language_detector_input_bucket: str,
    script_detection_result: str,
) -> str:
    if language_visibility_status == "visible":
        return "not_applicable"
    if text_extraction_attempted != "yes":
        return "metadata_missing"
    if text_source_present == "unknown":
        return "text_not_passed_to_visibility_detector"
    if text_source_present == "no" or text_extraction_result_bucket in {"none", "empty"}:
        return "no_text_available"
    if language_detector_attempted != "yes":
        return "detector_not_called"
    if script_detection_result in {"numeric_only", "symbol_only"}:
        return "numeric_or_symbol_only_text"
    if language_visibility_status == "unknown":
        return "detector_returned_unknown"
    if language_visibility_status == "incomplete":
        return "detector_returned_unknown"
    if language_detector_input_bucket in {"empty", "tiny"}:
        return "no_text_available"
    return "unknown"


def pdf_text_layer_detected(path: Path) -> str:
    if path.suffix.lower() != ".pdf":
        return "not_applicable"
    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(str(path))
        text_len = 0
        for index, page in enumerate(reader.pages):
            if index >= 3:
                break
            text_len += len(str(page.extract_text() or "").strip())
            if text_len >= 80:
                return "yes"
        return "no"
    except Exception:
        return "unknown"


def image_like_pdf(path: Path, pdf_text_layer: str, native_length_bucket: str) -> str:
    if path.suffix.lower() != ".pdf":
        return "not_applicable"
    if pdf_text_layer == "no" and native_length_bucket in {"none", "tiny", "short"}:
        return "yes"
    if pdf_text_layer == "yes":
        return "no"
    return "unknown"


def ocr_fallback_eligible(
    *,
    cyrillic_ocr_recommended: bool,
    image_like_pdf: str,
    language_visibility_status: str,
    native_text_length_bucket: str,
    extension: str,
) -> str:
    if extension != ".pdf":
        return "no"
    if cyrillic_ocr_recommended or image_like_pdf == "yes":
        return "yes"
    if language_visibility_status in {"missing", "incomplete", "unknown"} and native_text_length_bucket in {
        "none",
        "tiny",
        "short",
    }:
        return "unknown"
    return "no"


def ocr_fallback_not_triggered_reason(
    *,
    extension: str,
    ocr_fallback_executed: bool,
    ocr_fallback_eligible: str,
    pdf_text_layer_detected: str,
    image_like_pdf: str,
    native_text_length_bucket: str,
    language_visibility_status: str,
    error_bucket: str | None,
) -> str:
    if ocr_fallback_executed:
        return "fallback_executed"
    if error_bucket:
        return "extraction_error"
    if extension != ".pdf":
        return "routing_not_eligible"
    if image_like_pdf == "yes":
        return "image_like_pdf_but_not_routed_to_ocr"
    if pdf_text_layer_detected == "no":
        return "no_text_layer"
    if pdf_text_layer_detected == "yes" and native_text_length_bucket in {"none", "tiny", "short"}:
        return "text_layer_present_but_too_short"
    if ocr_fallback_eligible == "yes":
        return "unknown_reason"
    if language_visibility_status in {"unknown", "incomplete"}:
        return "language_visibility_unknown"
    if pdf_text_layer_detected == "unknown":
        return "unsupported_pdf_structure"
    return "routing_not_eligible"


def language_visibility_status(ocr_marker: dict[str, Any], extractor_result: dict[str, Any]) -> str:
    value = str(
        ocr_marker.get("language_text_visibility")
        or extractor_result.get("language_text_visibility")
        or ""
    ).lower()
    if value in {"visible", "recovered", "readable"}:
        return "visible"
    if value in {"incomplete", "not_recovered"}:
        return "incomplete"
    if value in {"not_applicable", "none"}:
        return "not_applicable"
    if not value or value == "unknown":
        return "unknown"
    return value


def cyrillic_visibility_status(ocr_marker: dict[str, Any], extractor_result: dict[str, Any]) -> str:
    if ocr_marker.get("ocr_gate_fallback_cyrillic_detected") is True:
        return "recovered"
    if extractor_result.get("ocr_gate_fallback_cyrillic_detected") is True:
        return "recovered"
    if ocr_marker.get("cyrillic_ocr_recommended") is True:
        return "missing"
    marker_visibility = str(ocr_marker.get("language_text_visibility") or "").lower()
    if marker_visibility == "visible":
        return "visible"
    if marker_visibility in {"incomplete", "not_recovered"}:
        return "missing"
    return "unknown"


def document_family_cue_count_bucket(family_diagnostic: dict[str, Any]) -> str:
    count = len(_safe_string_list(family_diagnostic.get("matched_family_cue_keys")))
    if count <= 0:
        return "none"
    if count == 1:
        return "one"
    if count <= 3:
        return "few"
    return "many"


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
