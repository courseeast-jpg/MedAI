from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from loguru import logger
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = ROOT / "test_data" / "final_batch_50"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "phase12_real_world_validation"
DEFAULT_PHASE13_REPORT_DIR = ROOT / "reports" / "phase13"
DEFAULT_PHASE15_REPORT_DIR = ROOT / "reports" / "phase15"
DEFAULT_PHASE21_ARTIFACT_PATH = ROOT / "artifacts" / "phase21" / "observability_metrics.json"
DEFAULT_PHASE21_REPORT_PATH = ROOT / "reports" / "phase21" / "observability_report.md"
DEFAULT_PHASE22_ARTIFACT_PATH = ROOT / "artifacts" / "phase22" / "confidence_calibration.json"
DEFAULT_PHASE22_REPORT_PATH = ROOT / "reports" / "phase22" / "accuracy_calibration_report.md"
DEFAULT_PHASE23_ARTIFACT_PATH = ROOT / "artifacts" / "phase23" / "routing_efficiency.json"
DEFAULT_PHASE23_REPORT_PATH = ROOT / "reports" / "phase23" / "routing_efficiency_report.md"
DEFAULT_PHASE24_ARTIFACT_PATH = ROOT / "artifacts" / "phase24" / "semantic_enrichment.json"
DEFAULT_PHASE24_REPORT_PATH = ROOT / "reports" / "phase24" / "semantic_enrichment_report.md"
DEFAULT_PHASE25_ARTIFACT_PATH = ROOT / "artifacts" / "phase25" / "medical_coding.json"
DEFAULT_PHASE25_REPORT_PATH = ROOT / "reports" / "phase25" / "medical_coding_report.md"
DEFAULT_PHASE26_ARTIFACT_PATH = ROOT / "artifacts" / "phase26" / "language_support.json"
DEFAULT_PHASE26_REPORT_PATH = ROOT / "reports" / "phase26" / "language_support_report.md"
ROUTING_DECISION_RE = re.compile(r"routing_decision=selected=([a-zA-Z0-9_]+)")
RETRY_DELAY_RE = re.compile(r"retry in (\d+(?:\.\d+)?)(?:s| seconds?)", re.IGNORECASE)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution.audit import StageAuditLogger
from execution.logging import AuditLogger
from execution.pipeline import ExecutionPipeline
from execution.review_queue import ReviewQueueWriter, read_review_queue
from mkb.sqlite_store import SQLiteStore
from monitoring.metrics_collector import collect_latest_run_metrics
from monitoring.observability import (
    write_phase21_outputs,
    write_phase22_outputs,
    write_phase23_outputs,
    write_phase24_outputs,
    write_phase25_outputs,
    write_phase26_outputs,
)


def is_external_quota_error(error: Exception | str) -> bool:
    message = str(error).lower()
    quota_markers = (
        "quota exceeded",
        "current quota",
        "rate limit",
        "rate-limit",
        "retry in",
        "generativelanguage.googleapis.com/generate_content",
        "429",
    )
    return any(marker in message for marker in quota_markers)


def list_pdf_documents(dataset_dir: Path, limit: int) -> list[Path]:
    pdfs = sorted(path for path in dataset_dir.glob("*.pdf") if path.is_file())
    if limit > 0:
        return pdfs[:limit]
    return pdfs


def build_validation_pipeline(runtime_dir: Path) -> tuple[ExecutionPipeline, SQLiteStore, dict[str, Any]]:
    runtime_dir.mkdir(parents=True, exist_ok=True)

    db_path = runtime_dir / "mkb.db"
    audit_path = runtime_dir / "execution_audit.jsonl"
    stage_audit_path = runtime_dir / "pipeline_stages.jsonl"
    review_queue_path = runtime_dir / "review_queue.jsonl"

    sql_store = SQLiteStore(db_path)
    pipeline = ExecutionPipeline(
        sql_store=sql_store,
        audit_logger=AuditLogger(path=audit_path),
        stage_audit_logger=StageAuditLogger(path=stage_audit_path),
        review_queue_path=review_queue_path,
    )
    component_state = {
        "sql_store": True,
        "vector_store": False,
        "quality_gate": False,
        "medication_gate": False,
        "governance_active": True,
        "review_queue_path": str(review_queue_path),
        "audit_log_path": str(audit_path),
        "stage_audit_log_path": str(stage_audit_path),
        "runtime_db_path": str(db_path),
    }
    return pipeline, sql_store, component_state


def parse_requested_route(notes: list[str], fallback: str | None = None) -> str | None:
    for note in notes:
        match = ROUTING_DECISION_RE.search(note)
        if match:
            return match.group(1)
    return fallback


def extract_retry_visibility(message: str) -> dict[str, Any]:
    match = RETRY_DELAY_RE.search(message)
    retry_delay_seconds = float(match.group(1)) if match else None
    retry_detected = retry_delay_seconds is not None or "retry" in message.lower()
    return {
        "retry_detected": retry_detected,
        "retry_delay_seconds": retry_delay_seconds,
        "retry_reason": "external_quota" if retry_detected and is_external_quota_error(message) else None,
    }


def summarize_document(
    pdf_path: Path,
    result,
    error: Exception | None = None,
    *,
    quota_safe: bool = False,
    processing_time_ms: float = 0.0,
) -> dict[str, Any]:
    if error is not None:
        retry_visibility = extract_retry_visibility(str(error))
        if retry_visibility["retry_detected"]:
            logger.info(
                "phase13 retry visibility document={} retry_detected=true retry_delay_seconds={} reason={}",
                pdf_path.name,
                retry_visibility["retry_delay_seconds"],
                retry_visibility["retry_reason"] or "unknown",
            )
        if quota_safe and is_external_quota_error(error):
            return {
                "document": pdf_path.name,
                "status": "external_quota_blocked",
                "error": str(error),
                "outcome": "external_quota_blocked",
                "validation_status": "skipped_external_quota",
                "extractor": None,
                "extractor_actual": None,
                "confidence": 0.0,
                "entity_count": 0,
                "written_count": 0,
                "queued_count": 0,
                "blocked_count": 0,
                "review_reasons": [],
                "notes": [],
                "processing_time_ms": round(processing_time_ms, 3),
                "requested_route": None,
                "retry_visibility": retry_visibility,
                "raw_confidence": None,
                "calibrated_confidence": None,
                "confidence_band": None,
                "calibration_reason": None,
                "route_mismatch_flag": False,
                "review_recommendation": "operator_retry_after_quota_reset",
                "intended_route": None,
                "actual_route": None,
                "fallback_reason": None,
                "estimated_cost_units": 0.0,
                "saved_cost_units": 0.0,
                "quota_block_avoided": False,
                "semantic_enrichment": None,
                "enrichment_applied": False,
                "negation_detected_count": 0,
                "temporal_detected_count": 0,
                "relationships_detected_count": 0,
                "medical_coding": None,
                "coding_applied": False,
                "coding_attempted_count": 0,
                "coding_success_count": 0,
                "coding_unmapped_count": 0,
                "coding_ambiguous_count": 0,
                "coding_skipped_count": 0,
                "language_support": None,
                "detected_language": "unknown",
                "language_confidence": 0.0,
                "script_detected": "unknown",
                "cyrillic_detected": False,
                "requires_ocr": False,
                "language_route_note": "metadata_only:language_signal_insufficient",
                "translation_status": "not_required",
                "language_support_status": "unknown_metadata_only",
            }
        return {
            "document": pdf_path.name,
            "status": "error",
            "error": str(error),
            "outcome": "error",
            "validation_status": "error",
            "extractor": None,
            "extractor_actual": None,
            "confidence": 0.0,
            "entity_count": 0,
            "written_count": 0,
            "queued_count": 0,
            "blocked_count": 0,
            "review_reasons": [],
            "notes": [],
            "processing_time_ms": round(processing_time_ms, 3),
            "requested_route": None,
            "retry_visibility": retry_visibility,
            "raw_confidence": None,
            "calibrated_confidence": None,
            "confidence_band": None,
            "calibration_reason": None,
            "route_mismatch_flag": False,
            "review_recommendation": None,
            "intended_route": None,
            "actual_route": None,
            "fallback_reason": None,
            "estimated_cost_units": 0.0,
            "saved_cost_units": 0.0,
            "quota_block_avoided": False,
            "semantic_enrichment": None,
            "enrichment_applied": False,
            "negation_detected_count": 0,
            "temporal_detected_count": 0,
            "relationships_detected_count": 0,
            "medical_coding": None,
            "coding_applied": False,
            "coding_attempted_count": 0,
            "coding_success_count": 0,
            "coding_unmapped_count": 0,
            "coding_ambiguous_count": 0,
            "coding_skipped_count": 0,
            "language_support": None,
            "detected_language": "unknown",
            "language_confidence": 0.0,
            "script_detected": "unknown",
            "cyrillic_detected": False,
            "requires_ocr": False,
            "language_route_note": "metadata_only:language_signal_insufficient",
            "translation_status": "not_required",
            "language_support_status": "unknown_metadata_only",
        }

    review_reasons: list[str] = []
    for item in result.validation_errors:
        code = item.get("code")
        if code:
            review_reasons.append(str(code))
    for record in result.queued_records + result.blocked_records:
        if record.status:
            review_reasons.append(str(record.status))
        if record.ddi_status:
            review_reasons.append(str(record.ddi_status))
    notes = list(result.notes)
    requested_route = (
        result.audit.get("requested_extractor_route")
        or result.extractor_result.get("requested_extractor_route")
        or parse_requested_route(notes)
    )
    semantic_enrichment = result.extractor_result.get("semantic_enrichment")
    if not isinstance(semantic_enrichment, dict):
        semantic_enrichment = None
    medical_coding = result.extractor_result.get("medical_coding")
    if not isinstance(medical_coding, dict):
        medical_coding = None
    language_support = result.extractor_result.get("language_support")
    if not isinstance(language_support, dict):
        language_support = None

    return {
        "document": pdf_path.name,
        "status": "processed",
        "error": None,
        "outcome": result.outcome,
        "validation_status": result.validation_status,
        "extractor_route": result.audit.get("extractor_route", result.extractor_result.get("extractor_route")),
        "extractor": result.audit.get("extractor", result.extractor_result.get("extractor")),
        "extractor_actual": result.audit.get("extractor_actual", result.extractor_result.get("actual_extractor")),
        "confidence": float(result.audit.get("confidence", result.extractor_result.get("confidence", 0.0))),
        "entity_count": int(result.audit.get("entity_count", len(result.extractor_result.get("entities", [])))),
        "written_count": result.written_count,
        "queued_count": result.queued_count,
        "blocked_count": len(result.blocked_records),
        "review_reasons": sorted(set(review_reasons)),
        "notes": notes,
        "processing_time_ms": round(processing_time_ms, 3),
        "requested_route": requested_route,
        "retry_visibility": {"retry_detected": False, "retry_delay_seconds": None, "retry_reason": None},
        "raw_confidence": result.audit.get("raw_confidence", result.extractor_result.get("raw_confidence")),
        "calibrated_confidence": result.audit.get(
            "calibrated_confidence",
            result.extractor_result.get("calibrated_confidence"),
        ),
        "confidence_band": result.audit.get("confidence_band", result.extractor_result.get("confidence_band")),
        "calibration_reason": result.audit.get(
            "calibration_reason",
            result.extractor_result.get("calibration_reason"),
        ),
        "route_mismatch_flag": bool(
            result.audit.get("route_mismatch_flag", result.extractor_result.get("route_mismatch_flag", False))
        ),
        "review_recommendation": result.audit.get(
            "review_recommendation",
            result.extractor_result.get("review_recommendation"),
        ),
        "intended_route": result.audit.get("intended_route", result.extractor_result.get("intended_route")),
        "actual_route": result.audit.get("actual_route", result.extractor_result.get("actual_route")),
        "fallback_reason": result.audit.get("fallback_reason", result.extractor_result.get("fallback_reason")),
        "estimated_cost_units": float(
            result.audit.get("estimated_cost_units", result.extractor_result.get("estimated_cost_units", 0.0))
        ),
        "saved_cost_units": float(
            result.audit.get("saved_cost_units", result.extractor_result.get("saved_cost_units", 0.0))
        ),
        "quota_block_avoided": bool(
            result.audit.get("quota_block_avoided", result.extractor_result.get("quota_block_avoided", False))
        ),
        "semantic_enrichment": semantic_enrichment,
        "enrichment_applied": bool(semantic_enrichment and semantic_enrichment.get("applied", False)),
        "negation_detected_count": int((semantic_enrichment or {}).get("negation_detected_count", 0)),
        "temporal_detected_count": int((semantic_enrichment or {}).get("temporal_detected_count", 0)),
        "relationships_detected_count": int((semantic_enrichment or {}).get("relationships_detected_count", 0)),
        "medical_coding": medical_coding,
        "coding_applied": bool(medical_coding and medical_coding.get("applied", False)),
        "coding_attempted_count": int((medical_coding or {}).get("coding_attempted_count", 0)),
        "coding_success_count": int((medical_coding or {}).get("coding_success_count", 0)),
        "coding_unmapped_count": int((medical_coding or {}).get("coding_unmapped_count", 0)),
        "coding_ambiguous_count": int((medical_coding or {}).get("coding_ambiguous_count", 0)),
        "coding_skipped_count": int((medical_coding or {}).get("coding_skipped_count", 0)),
        "language_support": language_support,
        "detected_language": str((language_support or {}).get("detected_language", "unknown")),
        "language_confidence": float((language_support or {}).get("language_confidence", 0.0)),
        "script_detected": str((language_support or {}).get("script_detected", "unknown")),
        "cyrillic_detected": bool((language_support or {}).get("cyrillic_detected", False)),
        "requires_ocr": bool((language_support or {}).get("requires_ocr", False)),
        "language_route_note": str((language_support or {}).get("language_route_note", "metadata_only:language_signal_insufficient")),
        "translation_status": str((language_support or {}).get("translation_status", "not_required")),
        "language_support_status": str((language_support or {}).get("language_support_status", "unknown_metadata_only")),
    }


def build_phase12_summary(
    *,
    dataset_dir: Path,
    requested_limit: int,
    documents: list[dict[str, Any]],
    runtime_counts: dict[str, int],
    component_state: dict[str, Any],
) -> dict[str, Any]:
    processed = [item for item in documents if item["status"] == "processed"]
    external_quota_blocked = [item for item in documents if item["status"] == "external_quota_blocked"]
    errors = [item for item in documents if item["status"] == "error"]
    coverage_count = len(processed) + len(external_quota_blocked)
    review_queue_path = component_state.get("review_queue_path")
    review_queue_items = read_review_queue(review_queue_path) if review_queue_path else []

    outcome_counts = Counter(item["outcome"] for item in processed)
    validation_counts = Counter(item["validation_status"] for item in processed)
    extractor_counts = Counter(
        item["extractor_actual"] or item["extractor"] or "unknown"
        for item in processed
    )
    avg_confidence = round(
        sum(float(item["confidence"]) for item in processed) / len(processed),
        3,
    ) if processed else 0.0

    review_reasons = Counter()
    enrichment_applied_count = 0
    negation_detected_count = 0
    temporal_detected_count = 0
    relationships_detected_count = 0
    coding_attempted_count = 0
    coding_success_count = 0
    coding_unmapped_count = 0
    coding_ambiguous_count = 0
    coding_skipped_count = 0
    language_detected_counts = Counter()
    cyrillic_detected_count = 0
    mixed_language_count = 0
    pending_translation_count = 0
    requires_ocr_count = 0
    language_unknown_count = 0
    for item in processed:
        for reason in item["review_reasons"]:
            review_reasons[reason] += 1
        enrichment_applied_count += int(bool(item.get("enrichment_applied", False)))
        negation_detected_count += int(item.get("negation_detected_count", 0))
        temporal_detected_count += int(item.get("temporal_detected_count", 0))
        relationships_detected_count += int(item.get("relationships_detected_count", 0))
        coding_attempted_count += int(item.get("coding_attempted_count", 0))
        coding_success_count += int(item.get("coding_success_count", 0))
        coding_unmapped_count += int(item.get("coding_unmapped_count", 0))
        coding_ambiguous_count += int(item.get("coding_ambiguous_count", 0))
        coding_skipped_count += int(item.get("coding_skipped_count", 0))
        detected_language = str(item.get("detected_language", "unknown"))
        language_detected_counts[detected_language] += 1
        cyrillic_detected_count += int(bool(item.get("cyrillic_detected", False)))
        mixed_language_count += int(detected_language == "mixed")
        pending_translation_count += int(str(item.get("translation_status", "")) == "pending_translation")
        requires_ocr_count += int(bool(item.get("requires_ocr", False)))
        language_unknown_count += int(detected_language == "unknown")

    written_document_count = outcome_counts.get("written", 0) + outcome_counts.get("written_with_review", 0)

    recommendations: list[str] = []
    if coverage_count < 10:
        recommendations.append("Expand the Phase 12 run to at least 10 documents to match the baseline target window.")
    if errors:
        recommendations.append("Investigate per-document processing errors before treating this run as representative.")
    if external_quota_blocked:
        recommendations.append("Some documents were skipped due to external API quota exhaustion; rerun later to complete the sample without changing pipeline behavior.")
    if sum(int(item["queued_count"]) for item in processed) > 0:
        recommendations.append("Inspect review-queued documents and classify whether queues are expected governance behavior or extraction misses.")
    if outcome_counts.get("blocked_ddi", 0) > 0:
        recommendations.append("Review medication safety blocks manually and confirm whether DDI blocks are clinically appropriate.")
    if avg_confidence < 0.5:
        recommendations.append("Average extraction confidence is low; review OCR quality and routed extractor behavior on the sampled PDFs.")
    if not recommendations:
        recommendations.append("Run a second Phase 12 batch from a different document mix to confirm stability beyond this initial sample.")

    fixed_seed = os.environ.get("PHASE19_FIXED_SEED")
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "phase": "Phase 12 Real-World Validation",
        "dataset_dir": str(dataset_dir),
        "requested_limit": requested_limit,
        "documents_selected": len(documents),
        "documents_processed": len(processed),
        "written": written_document_count,
        "queued_for_review": outcome_counts.get("queued_for_review", 0),
        "external_quota_blocked": len(external_quota_blocked),
        "hard_failures": len(errors),
        "documents_failed": len(errors),
        "target_window_met": 10 <= coverage_count <= 20,
        "phase12_started": coverage_count > 0,
        "run_passed": coverage_count > 0 and len(errors) == 0,
        "component_state": component_state,
        "review_queue": {
            "path": review_queue_path,
            "items": len(review_queue_items),
        },
        "aggregate": {
            "outcomes": dict(sorted(outcome_counts.items())),
            "validation_statuses": dict(sorted(validation_counts.items())),
            "extractors": dict(sorted(extractor_counts.items())),
            "avg_confidence": avg_confidence,
            "total_entities": sum(int(item["entity_count"]) for item in processed),
            "total_written": sum(int(item["written_count"]) for item in processed),
            "total_queued": sum(int(item["queued_count"]) for item in processed),
            "total_blocked": sum(int(item["blocked_count"]) for item in processed),
            "review_reasons": dict(sorted(review_reasons.items())),
            "enrichment_applied_count": enrichment_applied_count,
            "negation_detected_count": negation_detected_count,
            "temporal_detected_count": temporal_detected_count,
            "relationships_detected_count": relationships_detected_count,
            "coding_attempted_count": coding_attempted_count,
            "coding_success_count": coding_success_count,
            "coding_unmapped_count": coding_unmapped_count,
            "coding_ambiguous_count": coding_ambiguous_count,
            "coding_skipped_count": coding_skipped_count,
            "language_detected_counts": dict(sorted(language_detected_counts.items())),
            "cyrillic_detected_count": cyrillic_detected_count,
            "mixed_language_count": mixed_language_count,
            "pending_translation_count": pending_translation_count,
            "requires_ocr_count": requires_ocr_count,
            "language_unknown_count": language_unknown_count,
        },
        "mkb_counts": runtime_counts,
        "documents": documents,
        "determinism": {
            "mode": "deterministic_path",
            "seed": int(fixed_seed) if fixed_seed not in {None, ""} else None,
            "ordering": "sorted_pdf_listing",
        },
        "recommendations": recommendations,
    }


def build_phase13_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    documents = summary["documents"]
    processed = [item for item in documents if item["status"] == "processed"]
    all_with_timing = [item for item in documents if float(item.get("processing_time_ms", 0.0)) > 0.0]

    route_distribution = Counter(
        item.get("extractor_actual") or item.get("extractor") or "unknown"
        for item in processed
    )
    requested_route_distribution = Counter(
        item.get("requested_route") or "unknown"
        for item in documents
    )

    per_extractor = {}
    extractor_timings: dict[str, list[float]] = {}
    for item in processed:
        extractor = item.get("extractor_actual") or item.get("extractor") or "unknown"
        extractor_timings.setdefault(str(extractor), []).append(float(item.get("processing_time_ms", 0.0)))
    for extractor, timings in sorted(extractor_timings.items()):
        per_extractor[extractor] = {
            "documents": len(timings),
            "total_time_ms": round(sum(timings), 3),
            "avg_time_ms": round(sum(timings) / len(timings), 3) if timings else 0.0,
        }

    retry_events = []
    for item in documents:
        retry_visibility = item.get("retry_visibility", {})
        if retry_visibility.get("retry_detected"):
            retry_events.append({
                "document": item["document"],
                "status": item["status"],
                "retry_delay_seconds": retry_visibility.get("retry_delay_seconds"),
                "retry_reason": retry_visibility.get("retry_reason"),
                "error": item.get("error"),
            })

    total_pipeline_time_ms = round(sum(float(item.get("processing_time_ms", 0.0)) for item in all_with_timing), 3)
    document_times = [float(item.get("processing_time_ms", 0.0)) for item in all_with_timing]

    return {
        "generated_at": summary["generated_at"],
        "documents_processed": summary["documents_processed"],
        "written": summary["written"],
        "queued_for_review": summary["queued_for_review"],
        "external_quota_blocked": summary["external_quota_blocked"],
        "hard_failures": summary["hard_failures"],
        "avg_confidence": summary["aggregate"]["avg_confidence"],
        "route_distribution": dict(sorted(route_distribution.items())),
        "requested_route_distribution": dict(sorted(requested_route_distribution.items())),
        "timing": {
            "total_pipeline_time_ms": total_pipeline_time_ms,
            "avg_document_time_ms": round(sum(document_times) / len(document_times), 3) if document_times else 0.0,
            "max_document_time_ms": round(max(document_times), 3) if document_times else 0.0,
            "min_document_time_ms": round(min(document_times), 3) if document_times else 0.0,
            "per_extractor": per_extractor,
            "per_document": [
                {
                    "document": item["document"],
                    "status": item["status"],
                    "extractor_actual": item.get("extractor_actual") or item.get("extractor") or "unknown",
                    "processing_time_ms": round(float(item.get("processing_time_ms", 0.0)), 3),
                }
                for item in documents
            ],
        },
        "retries": {
            "retry_event_count": len(retry_events),
            "documents_with_retries": [item["document"] for item in retry_events],
            "total_suggested_backoff_seconds": round(
                sum(float(item["retry_delay_seconds"] or 0.0) for item in retry_events),
                3,
            ),
            "retry_events": retry_events,
        },
    }


def build_phase13_performance_summary(summary: dict[str, Any], metrics: dict[str, Any]) -> str:
    lines = [
        "# Phase 13 Performance Summary",
        "",
        f"- Generated at: {metrics['generated_at']}",
        f"- Documents processed: {summary['documents_processed']}/{summary['documents_selected']}",
        f"- Written: {summary['written']}",
        f"- Queued for review: {summary['queued_for_review']}",
        f"- External quota blocked: {summary['external_quota_blocked']}",
        f"- Hard failures: {summary['hard_failures']}",
        f"- Average confidence: {summary['aggregate']['avg_confidence']}",
        f"- Total pipeline time (ms): {metrics['timing']['total_pipeline_time_ms']}",
        f"- Average document time (ms): {metrics['timing']['avg_document_time_ms']}",
        "",
        "## Route Distribution",
        "",
        f"- Actual routes: {metrics['route_distribution']}",
        f"- Requested routes: {metrics['requested_route_distribution']}",
        "",
        "## Extractor Timing",
        "",
    ]
    if metrics["timing"]["per_extractor"]:
        for extractor, timing in metrics["timing"]["per_extractor"].items():
            lines.append(
                f"- `{extractor}` -> documents={timing['documents']} total_ms={timing['total_time_ms']} avg_ms={timing['avg_time_ms']}"
            )
    else:
        lines.append("- No processed extractor timing available.")

    lines.extend([
        "",
        "## Retry Visibility",
        "",
        f"- Retry events observed: {metrics['retries']['retry_event_count']}",
        f"- Total suggested backoff seconds: {metrics['retries']['total_suggested_backoff_seconds']}",
    ])
    if metrics["retries"]["retry_events"]:
        for event in metrics["retries"]["retry_events"]:
            lines.append(
                f"- `{event['document']}` -> status={event['status']} retry_delay_seconds={event['retry_delay_seconds']} reason={event['retry_reason']}"
            )
    else:
        lines.append("- No retry/backoff signals observed.")
    return "\n".join(lines) + "\n"


def write_phase13_reports(report_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics = build_phase13_metrics(summary)
    metrics_path = report_dir / "metrics_snapshot.json"
    performance_path = report_dir / "performance_summary.md"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    performance_path.write_text(build_phase13_performance_summary(summary, metrics), encoding="utf-8")
    return metrics


def build_phase15_aggregate(summary: dict[str, Any]) -> dict[str, Any]:
    documents = summary["documents"]
    processed = [item for item in documents if item["status"] == "processed"]
    external_quota_blocked = [item for item in documents if item["status"] == "external_quota_blocked"]
    hard_failures = [item for item in documents if item["status"] == "error"]
    rejection_patterns = Counter()
    for item in processed:
        for reason in item.get("review_reasons", []):
            rejection_patterns[reason] += 1

    return {
        "generated_at": summary["generated_at"],
        "phase": "Phase 15 Expanded Validation",
        "dataset_dir": summary["dataset_dir"],
        "documents_attempted": summary["documents_selected"],
        "documents_processed": len(processed),
        "documents_quota_blocked": len(external_quota_blocked),
        "written": summary["written"],
        "queued_for_review": summary["queued_for_review"],
        "hard_failures": len(hard_failures),
        "avg_confidence_processed_only": summary["aggregate"]["avg_confidence"],
        "route_distribution_actual": dict(sorted(summary["aggregate"]["extractors"].items())),
        "route_distribution_requested": dict(
            sorted(
                Counter(item.get("requested_route") or "unknown" for item in documents).items()
            )
        ),
        "top_rejection_patterns": [
            {"pattern": pattern, "count": count}
            for pattern, count in rejection_patterns.most_common(3)
        ],
        "counters": {
            "attempted_equals_processed_plus_quota_blocked_plus_hard_failures": (
                summary["documents_selected"] == len(processed) + len(external_quota_blocked) + len(hard_failures)
            ),
            "processed_equals_written_plus_queued_for_review_plus_other_processed_outcomes": (
                len(processed)
                == summary["written"]
                + summary["queued_for_review"]
                + sum(
                    1
                    for item in processed
                    if item["outcome"] not in {"written", "written_with_review", "queued_for_review"}
                )
            ),
        },
    }


def build_phase15_validation_summary(summary: dict[str, Any], aggregate: dict[str, Any]) -> str:
    lines = [
        "# Phase 15 Validation Summary",
        "",
        f"- Generated at: {aggregate['generated_at']}",
        f"- Dataset: `{aggregate['dataset_dir']}`",
        f"- Total documents attempted: {aggregate['documents_attempted']}",
        f"- Processed: {aggregate['documents_processed']}",
        f"- External quota blocked: {aggregate['documents_quota_blocked']}",
        f"- Written: {aggregate['written']}",
        f"- Queued for review: {aggregate['queued_for_review']}",
        f"- Hard failures: {aggregate['hard_failures']}",
        f"- Average confidence (processed only): {aggregate['avg_confidence_processed_only']}",
        "",
        "## Route Distribution",
        "",
        f"- Actual routes: {aggregate['route_distribution_actual']}",
        f"- Requested routes: {aggregate['route_distribution_requested']}",
        "",
        "## Rejection Patterns",
        "",
    ]
    if aggregate["top_rejection_patterns"]:
        for item in aggregate["top_rejection_patterns"]:
            lines.append(f"- `{item['pattern']}`: {item['count']}")
    else:
        lines.append("- No recurring rejection patterns observed in processed documents.")

    lines.extend([
        "",
        "## Validation Integrity",
        "",
        f"- Attempted counter check passed: {aggregate['counters']['attempted_equals_processed_plus_quota_blocked_plus_hard_failures']}",
        f"- Processed counter check passed: {aggregate['counters']['processed_equals_written_plus_queued_for_review_plus_other_processed_outcomes']}",
    ])
    return "\n".join(lines) + "\n"


def write_phase15_reports(report_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    aggregate = build_phase15_aggregate(summary)
    aggregate_path = report_dir / "validation_aggregate.json"
    summary_path = report_dir / "validation_summary.md"
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    summary_path.write_text(build_phase15_validation_summary(summary, aggregate), encoding="utf-8")
    return aggregate


def write_outputs(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "phase12_real_world_validation_summary.json"
    md_path = output_dir / "phase12_real_world_validation_summary.md"
    jsonl_path = output_dir / "phase12_real_world_validation_documents.jsonl"

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for item in summary["documents"]:
            handle.write(json.dumps(item, sort_keys=True) + "\n")

    md_lines = [
        "# Phase 12 Real-World Validation Summary",
        "",
        f"- Generated at: {summary['generated_at']}",
        f"- Dataset: `{summary['dataset_dir']}`",
        f"- Documents processed: {summary['documents_processed']}/{summary['documents_selected']}",
        f"- Written: {summary['written']}",
        f"- Queued for review: {summary['queued_for_review']}",
        f"- External quota blocked: {summary['external_quota_blocked']}",
        f"- Hard failures: {summary['hard_failures']}",
        f"- Review queue path: `{summary['review_queue']['path']}`",
        f"- Review queue items: {summary['review_queue']['items']}",
        f"- Target window met (10-20 docs): {summary['target_window_met']}",
        f"- Run passed: {summary['run_passed']}",
        "",
        "## Aggregate",
        "",
        f"- Outcomes: {summary['aggregate']['outcomes']}",
        f"- Validation statuses: {summary['aggregate']['validation_statuses']}",
        f"- Extractors: {summary['aggregate']['extractors']}",
        f"- Average confidence: {summary['aggregate']['avg_confidence']}",
        f"- Total entities: {summary['aggregate']['total_entities']}",
        f"- Total written: {summary['aggregate']['total_written']}",
        f"- Total queued: {summary['aggregate']['total_queued']}",
        f"- Total blocked: {summary['aggregate']['total_blocked']}",
        f"- Semantic enrichment applied: {summary['aggregate']['enrichment_applied_count']}",
        f"- Negation detected count: {summary['aggregate']['negation_detected_count']}",
        f"- Temporal detected count: {summary['aggregate']['temporal_detected_count']}",
        f"- Relationships detected count: {summary['aggregate']['relationships_detected_count']}",
        f"- Coding attempted count: {summary['aggregate']['coding_attempted_count']}",
        f"- Coding success count: {summary['aggregate']['coding_success_count']}",
        f"- Coding unmapped count: {summary['aggregate']['coding_unmapped_count']}",
        f"- Coding ambiguous count: {summary['aggregate']['coding_ambiguous_count']}",
        f"- Coding skipped count: {summary['aggregate']['coding_skipped_count']}",
        f"- Language detected counts: {summary['aggregate']['language_detected_counts']}",
        f"- Cyrillic detected count: {summary['aggregate']['cyrillic_detected_count']}",
        f"- Mixed language count: {summary['aggregate']['mixed_language_count']}",
        f"- Pending translation count: {summary['aggregate']['pending_translation_count']}",
        f"- Requires OCR count: {summary['aggregate']['requires_ocr_count']}",
        f"- Language unknown count: {summary['aggregate']['language_unknown_count']}",
        "",
        "## Runtime MKB",
        "",
        f"- Counts: {summary['mkb_counts']}",
        "",
        "## Component State",
        "",
    ]
    for key, value in summary["component_state"].items():
        md_lines.append(f"- {key}: {value}")
    md_lines.extend([
        "",
        "## Recommendations",
        "",
    ])
    md_lines.extend(f"- {item}" for item in summary["recommendations"])
    md_lines.extend([
        "",
        "## Documents",
        "",
    ])
    for item in summary["documents"]:
        md_lines.append(
            f"- `{item['document']}` -> status={item['status']} outcome={item['outcome']} "
            f"validation={item['validation_status']} confidence={item['confidence']}"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--specialty", default="general")
    parser.add_argument("--quota-safe", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    runtime_dir = output_dir / "runtime"

    if not dataset_dir.exists():
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")

    fixed_seed = os.environ.get("PHASE19_FIXED_SEED")
    if fixed_seed:
        logger.info("phase19 deterministic path fixed_seed={} ordering=sorted_pdf_listing", fixed_seed)
    else:
        logger.info("phase19 deterministic path seed=none ordering=sorted_pdf_listing")

    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)

    pdfs = list_pdf_documents(dataset_dir, args.limit)
    pipeline, sql_store, component_state = build_validation_pipeline(runtime_dir)
    review_queue = ReviewQueueWriter(Path(component_state["review_queue_path"]))

    documents: list[dict[str, Any]] = []
    for pdf_path in pdfs:
        started = time.perf_counter()
        try:
            result = pipeline.process_pdf(
                pdf_path,
                specialty=args.specialty,
                session_id=f"phase12-{pdf_path.stem}",
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            documents.append(summarize_document(pdf_path, result, processing_time_ms=elapsed_ms))
        except Exception as exc:  # pragma: no cover - defensive path for live validation
            elapsed_ms = (time.perf_counter() - started) * 1000
            document = summarize_document(
                pdf_path,
                None,
                error=exc,
                quota_safe=args.quota_safe,
                processing_time_ms=elapsed_ms,
            )
            if document["status"] == "external_quota_blocked":
                review_queue.append_external_quota_block(
                    run_id=f"phase12-{pdf_path.stem}",
                    document_id=pdf_path.name,
                    source_filename=pdf_path.name,
                    reason="external_quota_blocked",
                    recommended_action="operator_retry_after_quota_reset",
                    raw_evidence_path=str(pdf_path),
                    error=str(exc),
                    retry_visibility=document["retry_visibility"],
                )
            documents.append(document)

    summary = build_phase12_summary(
        dataset_dir=dataset_dir,
        requested_limit=args.limit,
        documents=documents,
        runtime_counts=sql_store.count_records(),
        component_state=component_state,
    )
    write_outputs(output_dir, summary)
    write_phase13_reports(DEFAULT_PHASE13_REPORT_DIR, summary)
    write_phase15_reports(DEFAULT_PHASE15_REPORT_DIR, summary)
    write_phase21_outputs(
        summary,
        artifact_path=DEFAULT_PHASE21_ARTIFACT_PATH,
        report_path=DEFAULT_PHASE21_REPORT_PATH,
    )
    write_phase22_outputs(
        summary,
        artifact_path=DEFAULT_PHASE22_ARTIFACT_PATH,
        report_path=DEFAULT_PHASE22_REPORT_PATH,
    )
    write_phase23_outputs(
        summary,
        artifact_path=DEFAULT_PHASE23_ARTIFACT_PATH,
        report_path=DEFAULT_PHASE23_REPORT_PATH,
    )
    write_phase24_outputs(
        summary,
        artifact_path=DEFAULT_PHASE24_ARTIFACT_PATH,
        report_path=DEFAULT_PHASE24_REPORT_PATH,
    )
    write_phase25_outputs(
        summary,
        artifact_path=DEFAULT_PHASE25_ARTIFACT_PATH,
        report_path=DEFAULT_PHASE25_REPORT_PATH,
    )
    write_phase26_outputs(
        summary,
        artifact_path=DEFAULT_PHASE26_ARTIFACT_PATH,
        report_path=DEFAULT_PHASE26_REPORT_PATH,
    )
    collect_latest_run_metrics()

    print(json.dumps(summary, indent=2))
    return 0 if summary["run_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
