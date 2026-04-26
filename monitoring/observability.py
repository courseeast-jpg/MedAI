from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import EXTRACTION_ACCEPT_THRESHOLD
from execution.review_queue import read_review_queue


ROOT = Path(__file__).resolve().parent.parent
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
DEFAULT_PHASE27_ARTIFACT_PATH = ROOT / "artifacts" / "phase27" / "runtime_controls.json"
DEFAULT_PHASE27_REPORT_PATH = ROOT / "reports" / "phase27" / "production_hardening_report.md"
DEFAULT_PHASE28_ARTIFACT_PATH = ROOT / "artifacts" / "phase28" / "production_mode.json"
DEFAULT_PHASE28_REPORT_PATH = ROOT / "reports" / "phase28" / "production_readiness_report.md"


def _load_jsonl(path: Path | str | None) -> list[dict[str, Any]]:
    if path in {None, ""}:
        return []
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return []
    return [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def build_stage_duration_metrics(stage_events: list[dict[str, Any]]) -> dict[str, Any]:
    per_stage_record_times: dict[str, dict[str, list[datetime]]] = defaultdict(lambda: defaultdict(list))
    for event in stage_events:
        timestamp = _parse_timestamp(str(event.get("timestamp", "")))
        stage = str(event.get("stage", "unknown"))
        record_id = str(event.get("record_id", "unknown"))
        if timestamp is None:
            continue
        per_stage_record_times[stage][record_id].append(timestamp)

    stage_metrics: dict[str, Any] = {}
    for stage, record_times in sorted(per_stage_record_times.items()):
        durations_ms: list[float] = []
        for timestamps in record_times.values():
            start = min(timestamps)
            end = max(timestamps)
            durations_ms.append(round((end - start).total_seconds() * 1000.0, 3))
        stage_metrics[stage] = {
            "records": len(record_times),
            "events": sum(len(timestamps) for timestamps in record_times.values()),
            "total_duration_ms": round(sum(durations_ms), 3),
            "avg_duration_ms": round(sum(durations_ms) / len(durations_ms), 3) if durations_ms else 0.0,
            "max_duration_ms": round(max(durations_ms), 3) if durations_ms else 0.0,
        }
    return stage_metrics


def build_phase21_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    documents = summary.get("documents", [])
    processed = [item for item in documents if item.get("status") == "processed"]
    review_queue_path = summary.get("review_queue", {}).get("path") or summary.get("component_state", {}).get("review_queue_path")
    review_queue_items = read_review_queue(review_queue_path)
    stage_audit_path = summary.get("component_state", {}).get("stage_audit_log_path")
    stage_events = _load_jsonl(stage_audit_path)

    extractor_route_counts = Counter()
    extractor_actual_counts = Counter()
    route_mismatch_count = 0
    low_confidence_count = 0

    for item in processed:
        route_value = str(item.get("extractor_route") or "unknown")
        actual_value = str(item.get("extractor_actual") or item.get("extractor") or "unknown")
        requested_value = str(item.get("requested_route") or "unknown")
        confidence_band = str(item.get("confidence_band") or "")

        extractor_route_counts[route_value] += 1
        extractor_actual_counts[actual_value] += 1
        if bool(item.get("route_mismatch_flag", False)) or (
            requested_value != "unknown" and requested_value != actual_value
        ):
            route_mismatch_count += 1
        if confidence_band in {"review", "reject"} or float(item.get("confidence", 0.0)) < EXTRACTION_ACCEPT_THRESHOLD:
            low_confidence_count += 1

    review_queue_categories = Counter(
        str(item.get("queue_category", "unknown"))
        for item in review_queue_items
    )

    quota_safe_block_count = sum(
        1 for item in documents
        if item.get("status") == "external_quota_blocked"
    )
    enrichment_applied_count = sum(int(bool(item.get("enrichment_applied", False))) for item in processed)
    negation_detected_count = sum(int(item.get("negation_detected_count", 0)) for item in processed)
    temporal_detected_count = sum(int(item.get("temporal_detected_count", 0)) for item in processed)
    relationships_detected_count = sum(int(item.get("relationships_detected_count", 0)) for item in processed)
    coding_attempted_count = sum(int(item.get("coding_attempted_count", 0)) for item in processed)
    coding_success_count = sum(int(item.get("coding_success_count", 0)) for item in processed)
    coding_unmapped_count = sum(int(item.get("coding_unmapped_count", 0)) for item in processed)
    coding_ambiguous_count = sum(int(item.get("coding_ambiguous_count", 0)) for item in processed)
    coding_skipped_count = sum(int(item.get("coding_skipped_count", 0)) for item in processed)
    language_detected_counts = Counter(str(item.get("detected_language", "unknown")) for item in processed)
    cyrillic_detected_count = sum(int(bool(item.get("cyrillic_detected", False))) for item in processed)
    mixed_language_count = sum(int(str(item.get("detected_language", "")) == "mixed") for item in processed)
    pending_translation_count = sum(int(str(item.get("translation_status", "")) == "pending_translation") for item in processed)
    requires_ocr_count = sum(int(bool(item.get("requires_ocr", False))) for item in processed)
    language_unknown_count = sum(int(str(item.get("detected_language", "")) == "unknown") for item in processed)
    runtime_controls = summary.get("runtime_controls", {})
    production_mode = summary.get("production_mode", {})

    return {
        "generated_at": summary.get("generated_at"),
        "phase": "Phase 21 Observability & Metrics Tightening",
        "dataset_dir": summary.get("dataset_dir"),
        "determinism": summary.get("determinism", {}),
        "attempted_documents": int(summary.get("documents_selected", 0)),
        "processed_documents": int(summary.get("documents_processed", 0)),
        "written_documents": int(summary.get("written", 0)),
        "queued_for_review_documents": int(summary.get("queued_for_review", 0)),
        "review_queue_items": len(review_queue_items),
        "external_quota_blocked": int(summary.get("external_quota_blocked", 0)),
        "hard_failures": int(summary.get("hard_failures", 0)),
        "average_confidence": float(summary.get("aggregate", {}).get("avg_confidence", 0.0)),
        "extractor_route_counts": dict(sorted(extractor_route_counts.items())),
        "extractor_actual_counts": dict(sorted(extractor_actual_counts.items())),
        "route_mismatch_count": route_mismatch_count,
        "low_confidence_count": low_confidence_count,
        "quota_safe_block_count": quota_safe_block_count,
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
        "run_lock_acquired": bool(runtime_controls.get("run_lock_acquired", False)),
        "run_lock_released": bool(runtime_controls.get("run_lock_released", False)),
        "stale_lock_recovered": bool(runtime_controls.get("stale_lock_recovered", False)),
        "retry_eligible_count": int(runtime_controls.get("retry_eligible_count", 0)),
        "non_retryable_failure_count": int(runtime_controls.get("non_retryable_failure_count", 0)),
        "timeout_count": int(runtime_controls.get("timeout_count", 0)),
        "cleanup_completed": bool(runtime_controls.get("cleanup_completed", False)),
        "production_mode": str(production_mode.get("production_mode", "OFF")),
        "production_gate_passed": bool(production_mode.get("production_gate_passed", True)),
        "production_gate_failed_reason": production_mode.get("production_gate_failed_reason"),
        "dry_run_executed": bool(production_mode.get("dry_run_executed", False)),
        "controlled_run_limit_applied": bool(production_mode.get("controlled_run_limit_applied", False)),
        "run_blocked_by_gate": bool(production_mode.get("run_blocked_by_gate", False)),
        "review_queue_category_counts": dict(sorted(review_queue_categories.items())),
        "per_stage_duration_ms": build_stage_duration_metrics(stage_events),
        "paths": {
            "review_queue_path": review_queue_path,
            "stage_audit_path": stage_audit_path,
        },
    }


def build_phase21_report(metrics: dict[str, Any]) -> str:
    lines = [
        "# Phase 21 Observability Report",
        "",
        f"- Generated at: `{metrics['generated_at']}`",
        f"- Dataset: `{metrics['dataset_dir']}`",
        f"- Attempted documents: `{metrics['attempted_documents']}`",
        f"- Processed documents: `{metrics['processed_documents']}`",
        f"- Written documents: `{metrics['written_documents']}`",
        f"- Queued for review documents: `{metrics['queued_for_review_documents']}`",
        f"- Review queue items: `{metrics['review_queue_items']}`",
        f"- External quota blocked: `{metrics['external_quota_blocked']}`",
        f"- Hard failures: `{metrics['hard_failures']}`",
        f"- Average confidence: `{metrics['average_confidence']}`",
        f"- Route mismatch count: `{metrics['route_mismatch_count']}`",
        f"- Low-confidence count: `{metrics['low_confidence_count']}`",
        f"- Quota-safe block count: `{metrics['quota_safe_block_count']}`",
        f"- Enrichment applied count: `{metrics['enrichment_applied_count']}`",
        f"- Negation detected count: `{metrics['negation_detected_count']}`",
        f"- Temporal detected count: `{metrics['temporal_detected_count']}`",
        f"- Relationships detected count: `{metrics['relationships_detected_count']}`",
        f"- Coding attempted count: `{metrics['coding_attempted_count']}`",
        f"- Coding success count: `{metrics['coding_success_count']}`",
        f"- Coding unmapped count: `{metrics['coding_unmapped_count']}`",
        f"- Coding ambiguous count: `{metrics['coding_ambiguous_count']}`",
        f"- Coding skipped count: `{metrics['coding_skipped_count']}`",
        f"- Language detected counts: `{metrics['language_detected_counts']}`",
        f"- Cyrillic detected count: `{metrics['cyrillic_detected_count']}`",
        f"- Mixed language count: `{metrics['mixed_language_count']}`",
        f"- Pending translation count: `{metrics['pending_translation_count']}`",
        f"- Requires OCR count: `{metrics['requires_ocr_count']}`",
        f"- Language unknown count: `{metrics['language_unknown_count']}`",
        f"- Run lock acquired: `{metrics['run_lock_acquired']}`",
        f"- Run lock released: `{metrics['run_lock_released']}`",
        f"- Stale lock recovered: `{metrics['stale_lock_recovered']}`",
        f"- Retry eligible count: `{metrics['retry_eligible_count']}`",
        f"- Non-retryable failure count: `{metrics['non_retryable_failure_count']}`",
        f"- Timeout count: `{metrics['timeout_count']}`",
        f"- Cleanup completed: `{metrics['cleanup_completed']}`",
        f"- Production mode: `{metrics['production_mode']}`",
        f"- Production gate passed: `{metrics['production_gate_passed']}`",
        f"- Production gate failed reason: `{metrics['production_gate_failed_reason']}`",
        f"- Dry run executed: `{metrics['dry_run_executed']}`",
        f"- Controlled run limit applied: `{metrics['controlled_run_limit_applied']}`",
        f"- Run blocked by gate: `{metrics['run_blocked_by_gate']}`",
        "",
        "## Route Counts",
        "",
        f"- Extractor route counts: `{metrics['extractor_route_counts']}`",
        f"- Extractor actual counts: `{metrics['extractor_actual_counts']}`",
        "",
        "## Review Queue",
        "",
        f"- Review queue category counts: `{metrics['review_queue_category_counts']}`",
        f"- Review queue path: `{metrics['paths']['review_queue_path']}`",
        "",
        "## Stage Durations",
        "",
    ]
    if metrics["per_stage_duration_ms"]:
        for stage, values in metrics["per_stage_duration_ms"].items():
            lines.append(
                f"- `{stage}` -> records={values['records']} events={values['events']} total_ms={values['total_duration_ms']} avg_ms={values['avg_duration_ms']} max_ms={values['max_duration_ms']}"
            )
    else:
        lines.append("- No stage timing data available.")
    lines.extend([
        "",
        "## Stability Guardrails",
        "",
        f"- Determinism: `{metrics['determinism']}`",
        "- External quota blocks are counted separately from hard failures.",
        "- Review queue counts are derived from the normalized JSONL queue written during validation.",
    ])
    return "\n".join(lines) + "\n"


def write_phase21_outputs(
    summary: dict[str, Any],
    *,
    artifact_path: Path = DEFAULT_PHASE21_ARTIFACT_PATH,
    report_path: Path = DEFAULT_PHASE21_REPORT_PATH,
) -> dict[str, Any]:
    metrics = build_phase21_metrics(summary)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report_path.write_text(build_phase21_report(metrics), encoding="utf-8")
    return metrics


def build_phase22_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    documents = summary.get("documents", [])
    processed = [item for item in documents if item.get("status") == "processed"]

    confidence_band_counts = Counter()
    calibration_reason_counts = Counter()
    review_recommendation_counts = Counter()
    extractor_route_counts = Counter()
    extractor_actual_counts = Counter()
    route_mismatch_count = 0

    calibration_documents: list[dict[str, Any]] = []
    for item in processed:
        route_value = str(item.get("extractor_route") or "unknown")
        actual_value = str(item.get("extractor_actual") or item.get("extractor") or "unknown")
        confidence_band = str(item.get("confidence_band") or "unknown")
        calibration_reason = str(item.get("calibration_reason") or "unknown")
        review_recommendation = str(item.get("review_recommendation") or "unknown")
        route_mismatch_flag = bool(item.get("route_mismatch_flag", False))

        confidence_band_counts[confidence_band] += 1
        calibration_reason_counts[calibration_reason] += 1
        review_recommendation_counts[review_recommendation] += 1
        extractor_route_counts[route_value] += 1
        extractor_actual_counts[actual_value] += 1
        if route_mismatch_flag:
            route_mismatch_count += 1

        calibration_documents.append({
            "document": item.get("document"),
            "outcome": item.get("outcome"),
            "validation_status": item.get("validation_status"),
            "raw_confidence": float(item.get("raw_confidence", item.get("confidence", 0.0))),
            "calibrated_confidence": float(item.get("calibrated_confidence", item.get("confidence", 0.0))),
            "confidence_band": confidence_band,
            "calibration_reason": calibration_reason,
            "extractor_route": route_value,
            "extractor_actual": actual_value,
            "route_mismatch_flag": route_mismatch_flag,
            "review_recommendation": review_recommendation,
        })

    raw_confidences = [float(item["raw_confidence"]) for item in calibration_documents]
    calibrated_confidences = [float(item["calibrated_confidence"]) for item in calibration_documents]

    return {
        "generated_at": summary.get("generated_at"),
        "phase": "Phase 22 Accuracy Improvement / Confidence Calibration",
        "dataset_dir": summary.get("dataset_dir"),
        "determinism": summary.get("determinism", {}),
        "attempted_documents": int(summary.get("documents_selected", 0)),
        "processed_documents": int(summary.get("documents_processed", 0)),
        "written_documents": int(summary.get("written", 0)),
        "queued_for_review_documents": int(summary.get("queued_for_review", 0)),
        "external_quota_blocked": int(summary.get("external_quota_blocked", 0)),
        "hard_failures": int(summary.get("hard_failures", 0)),
        "average_raw_confidence": round(sum(raw_confidences) / len(raw_confidences), 3) if raw_confidences else 0.0,
        "average_calibrated_confidence": (
            round(sum(calibrated_confidences) / len(calibrated_confidences), 3) if calibrated_confidences else 0.0
        ),
        "confidence_band_counts": dict(sorted(confidence_band_counts.items())),
        "calibration_reason_counts": dict(sorted(calibration_reason_counts.items())),
        "review_recommendation_counts": dict(sorted(review_recommendation_counts.items())),
        "extractor_route_counts": dict(sorted(extractor_route_counts.items())),
        "extractor_actual_counts": dict(sorted(extractor_actual_counts.items())),
        "route_mismatch_count": route_mismatch_count,
        "documents": calibration_documents,
    }


def build_phase22_report(metrics: dict[str, Any]) -> str:
    lines = [
        "# Phase 22 Accuracy Calibration Report",
        "",
        f"- Generated at: `{metrics['generated_at']}`",
        f"- Dataset: `{metrics['dataset_dir']}`",
        f"- Attempted documents: `{metrics['attempted_documents']}`",
        f"- Processed documents: `{metrics['processed_documents']}`",
        f"- Written documents: `{metrics['written_documents']}`",
        f"- Queued for review documents: `{metrics['queued_for_review_documents']}`",
        f"- External quota blocked: `{metrics['external_quota_blocked']}`",
        f"- Hard failures: `{metrics['hard_failures']}`",
        f"- Average raw confidence: `{metrics['average_raw_confidence']}`",
        f"- Average calibrated confidence: `{metrics['average_calibrated_confidence']}`",
        f"- Route mismatch count: `{metrics['route_mismatch_count']}`",
        "",
        "## Calibration Summary",
        "",
        f"- Confidence band counts: `{metrics['confidence_band_counts']}`",
        f"- Calibration reason counts: `{metrics['calibration_reason_counts']}`",
        f"- Review recommendation counts: `{metrics['review_recommendation_counts']}`",
        f"- Extractor route counts: `{metrics['extractor_route_counts']}`",
        f"- Extractor actual counts: `{metrics['extractor_actual_counts']}`",
        "",
        "## Document Audit",
        "",
    ]
    if metrics["documents"]:
        for item in metrics["documents"]:
            lines.append(
                f"- `{item['document']}` -> band={item['confidence_band']} raw={item['raw_confidence']} "
                f"calibrated={item['calibrated_confidence']} recommendation={item['review_recommendation']} "
                f"route_mismatch={item['route_mismatch_flag']}"
            )
    else:
        lines.append("- No processed documents available for calibration audit.")
    lines.extend([
        "",
        "## Stability Guardrails",
        "",
        f"- Determinism: `{metrics['determinism']}`",
        "- Review-band confidence remains observable through the review queue and audit fields.",
        "- Reject-band confidence is not written as accepted output.",
    ])
    return "\n".join(lines) + "\n"


def write_phase22_outputs(
    summary: dict[str, Any],
    *,
    artifact_path: Path = DEFAULT_PHASE22_ARTIFACT_PATH,
    report_path: Path = DEFAULT_PHASE22_REPORT_PATH,
) -> dict[str, Any]:
    metrics = build_phase22_metrics(summary)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report_path.write_text(build_phase22_report(metrics), encoding="utf-8")
    return metrics


def _normalized_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "None", "null"}:
        return None
    return text


def build_phase23_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    documents = summary.get("documents", [])
    processed = [item for item in documents if item.get("status") == "processed"]

    intended_route_counts = Counter()
    actual_route_counts = Counter()
    confidence_band_counts = Counter()
    review_recommendation_counts = Counter()
    fallback_reason_counts = Counter()
    total_estimated_cost_units = 0.0
    total_saved_cost_units = 0.0
    route_mismatch_count = 0
    quota_block_avoided_count = 0

    routing_documents: list[dict[str, Any]] = []
    for item in processed:
        intended_route = str(item.get("intended_route") or item.get("requested_route") or "unknown")
        actual_route = str(item.get("actual_route") or item.get("extractor_actual") or item.get("extractor") or "unknown")
        fallback_reason = _normalized_optional_text(item.get("fallback_reason"))
        route_mismatch_flag = bool(item.get("route_mismatch_flag", False))
        estimated_cost_units = round(float(item.get("estimated_cost_units", 0.0)), 5)
        saved_cost_units = round(float(item.get("saved_cost_units", 0.0)), 5)
        quota_block_avoided = bool(item.get("quota_block_avoided", False))
        confidence_band = str(item.get("confidence_band") or "unknown")
        review_recommendation = str(item.get("review_recommendation") or "unknown")

        intended_route_counts[intended_route] += 1
        actual_route_counts[actual_route] += 1
        confidence_band_counts[confidence_band] += 1
        review_recommendation_counts[review_recommendation] += 1
        if fallback_reason:
            fallback_reason_counts[str(fallback_reason)] += 1
        if route_mismatch_flag:
            route_mismatch_count += 1
        if quota_block_avoided:
            quota_block_avoided_count += 1
        total_estimated_cost_units += estimated_cost_units
        total_saved_cost_units += saved_cost_units

        routing_documents.append({
            "document": item.get("document"),
            "outcome": item.get("outcome"),
            "validation_status": item.get("validation_status"),
            "intended_route": intended_route,
            "actual_route": actual_route,
            "fallback_reason": fallback_reason,
            "route_mismatch_flag": route_mismatch_flag,
            "estimated_cost_units": estimated_cost_units,
            "saved_cost_units": saved_cost_units,
            "quota_block_avoided": quota_block_avoided,
            "confidence_band": confidence_band,
            "review_recommendation": review_recommendation,
        })

    return {
        "generated_at": summary.get("generated_at"),
        "phase": "Phase 23 Cost Optimization / Tier Routing Efficiency",
        "dataset_dir": summary.get("dataset_dir"),
        "determinism": summary.get("determinism", {}),
        "attempted_documents": int(summary.get("documents_selected", 0)),
        "processed_documents": int(summary.get("documents_processed", 0)),
        "written_documents": int(summary.get("written", 0)),
        "queued_for_review_documents": int(summary.get("queued_for_review", 0)),
        "external_quota_blocked": int(summary.get("external_quota_blocked", 0)),
        "hard_failures": int(summary.get("hard_failures", 0)),
        "intended_route_counts": dict(sorted(intended_route_counts.items())),
        "actual_route_counts": dict(sorted(actual_route_counts.items())),
        "confidence_band_counts": dict(sorted(confidence_band_counts.items())),
        "review_recommendation_counts": dict(sorted(review_recommendation_counts.items())),
        "fallback_reason_counts": dict(sorted(fallback_reason_counts.items())),
        "route_mismatch_count": route_mismatch_count,
        "quota_block_avoided_count": quota_block_avoided_count,
        "total_estimated_cost_units": round(total_estimated_cost_units, 5),
        "total_saved_cost_units": round(total_saved_cost_units, 5),
        "documents": routing_documents,
    }


def build_phase23_report(metrics: dict[str, Any]) -> str:
    lines = [
        "# Phase 23 Routing Efficiency Report",
        "",
        f"- Generated at: `{metrics['generated_at']}`",
        f"- Dataset: `{metrics['dataset_dir']}`",
        f"- Attempted documents: `{metrics['attempted_documents']}`",
        f"- Processed documents: `{metrics['processed_documents']}`",
        f"- Written documents: `{metrics['written_documents']}`",
        f"- Queued for review documents: `{metrics['queued_for_review_documents']}`",
        f"- External quota blocked: `{metrics['external_quota_blocked']}`",
        f"- Hard failures: `{metrics['hard_failures']}`",
        f"- Route mismatch count: `{metrics['route_mismatch_count']}`",
        f"- Quota block avoided count: `{metrics['quota_block_avoided_count']}`",
        f"- Total estimated cost units: `{metrics['total_estimated_cost_units']}`",
        f"- Total saved cost units: `{metrics['total_saved_cost_units']}`",
        "",
        "## Route Summary",
        "",
        f"- Intended route counts: `{metrics['intended_route_counts']}`",
        f"- Actual route counts: `{metrics['actual_route_counts']}`",
        f"- Confidence band counts: `{metrics['confidence_band_counts']}`",
        f"- Review recommendation counts: `{metrics['review_recommendation_counts']}`",
        f"- Fallback reason counts: `{metrics['fallback_reason_counts']}`",
        "",
        "## Document Audit",
        "",
    ]
    if metrics["documents"]:
        for item in metrics["documents"]:
            lines.append(
                f"- `{item['document']}` -> intended={item['intended_route']} actual={item['actual_route']} "
                f"saved_cost={item['saved_cost_units']} quota_avoided={item['quota_block_avoided']} "
                f"band={item['confidence_band']} recommendation={item['review_recommendation']}"
            )
    else:
        lines.append("- No processed documents available for routing-efficiency audit.")
    lines.extend([
        "",
        "## Stability Guardrails",
        "",
        f"- Determinism: `{metrics['determinism']}`",
        "- Review-band documents remain review-visible and are not silently accepted.",
        "- Quota-safe blocks remain separate from hard failures.",
    ])
    return "\n".join(lines) + "\n"


def write_phase23_outputs(
    summary: dict[str, Any],
    *,
    artifact_path: Path = DEFAULT_PHASE23_ARTIFACT_PATH,
    report_path: Path = DEFAULT_PHASE23_REPORT_PATH,
) -> dict[str, Any]:
    metrics = build_phase23_metrics(summary)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report_path.write_text(build_phase23_report(metrics), encoding="utf-8")
    return metrics


def build_phase24_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    documents = summary.get("documents", [])
    processed = [item for item in documents if item.get("status") == "processed"]

    enrichment_applied_count = 0
    negation_detected_count = 0
    temporal_detected_count = 0
    relationships_detected_count = 0
    enriched_documents: list[dict[str, Any]] = []

    for item in processed:
        semantic_enrichment = item.get("semantic_enrichment")
        applied = bool(item.get("enrichment_applied", False))
        if applied:
            enrichment_applied_count += 1
        negation_detected_count += int(item.get("negation_detected_count", 0))
        temporal_detected_count += int(item.get("temporal_detected_count", 0))
        relationships_detected_count += int(item.get("relationships_detected_count", 0))
        enriched_documents.append({
            "document": item.get("document"),
            "outcome": item.get("outcome"),
            "validation_status": item.get("validation_status"),
            "confidence": float(item.get("confidence", 0.0)),
            "confidence_band": str(item.get("confidence_band") or "unknown"),
            "enrichment_applied": applied,
            "negation_detected_count": int(item.get("negation_detected_count", 0)),
            "temporal_detected_count": int(item.get("temporal_detected_count", 0)),
            "relationships_detected_count": int(item.get("relationships_detected_count", 0)),
            "semantic_enrichment": semantic_enrichment,
        })

    return {
        "generated_at": summary.get("generated_at"),
        "phase": "Phase 24 Semantic Enrichment (non-destructive)",
        "dataset_dir": summary.get("dataset_dir"),
        "determinism": summary.get("determinism", {}),
        "attempted_documents": int(summary.get("documents_selected", 0)),
        "processed_documents": int(summary.get("documents_processed", 0)),
        "written_documents": int(summary.get("written", 0)),
        "queued_for_review_documents": int(summary.get("queued_for_review", 0)),
        "external_quota_blocked": int(summary.get("external_quota_blocked", 0)),
        "hard_failures": int(summary.get("hard_failures", 0)),
        "enrichment_applied_count": enrichment_applied_count,
        "negation_detected_count": negation_detected_count,
        "temporal_detected_count": temporal_detected_count,
        "relationships_detected_count": relationships_detected_count,
        "documents": enriched_documents,
    }


def build_phase24_report(metrics: dict[str, Any]) -> str:
    lines = [
        "# Phase 24 Semantic Enrichment Report",
        "",
        f"- Generated at: `{metrics['generated_at']}`",
        f"- Dataset: `{metrics['dataset_dir']}`",
        f"- Attempted documents: `{metrics['attempted_documents']}`",
        f"- Processed documents: `{metrics['processed_documents']}`",
        f"- Written documents: `{metrics['written_documents']}`",
        f"- Queued for review documents: `{metrics['queued_for_review_documents']}`",
        f"- External quota blocked: `{metrics['external_quota_blocked']}`",
        f"- Hard failures: `{metrics['hard_failures']}`",
        f"- Enrichment applied count: `{metrics['enrichment_applied_count']}`",
        f"- Negation detected count: `{metrics['negation_detected_count']}`",
        f"- Temporal detected count: `{metrics['temporal_detected_count']}`",
        f"- Relationships detected count: `{metrics['relationships_detected_count']}`",
        "",
        "## Document Audit",
        "",
    ]
    if metrics["documents"]:
        for item in metrics["documents"]:
            lines.append(
                f"- `{item['document']}` -> applied={item['enrichment_applied']} "
                f"band={item['confidence_band']} negation={item['negation_detected_count']} "
                f"temporal={item['temporal_detected_count']} relationships={item['relationships_detected_count']}"
            )
    else:
        lines.append("- No processed documents available for semantic enrichment audit.")
    lines.extend([
        "",
        "## Non-Destructive Guardrails",
        "",
        f"- Determinism: `{metrics['determinism']}`",
        "- Semantic enrichment is additive metadata only and does not alter confidence, routing, or review decisions.",
        "- Reject-band outputs are not semantically enriched.",
        "- The final Phase 24 deterministic baseline preserves `long_noisy_03.pdf` on a protected phi3 non-accepted path so live Gemini availability cannot flip the document between accepted and queued outcomes across reruns.",
        (
            f"- Final chosen aggregate for this generated report: written=`{metrics['written_documents']}` "
            f"queued_for_review=`{metrics['queued_for_review_documents']}` "
            f"external_quota_blocked=`{metrics['external_quota_blocked']}` "
            f"hard_failures=`{metrics['hard_failures']}`."
        ),
    ])
    return "\n".join(lines) + "\n"


def write_phase24_outputs(
    summary: dict[str, Any],
    *,
    artifact_path: Path = DEFAULT_PHASE24_ARTIFACT_PATH,
    report_path: Path = DEFAULT_PHASE24_REPORT_PATH,
) -> dict[str, Any]:
    metrics = build_phase24_metrics(summary)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report_path.write_text(build_phase24_report(metrics), encoding="utf-8")
    return metrics


def build_phase25_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    documents = summary.get("documents", [])
    processed = [item for item in documents if item.get("status") == "processed"]

    coding_attempted_count = 0
    coding_success_count = 0
    coding_unmapped_count = 0
    coding_ambiguous_count = 0
    coding_skipped_count = 0
    coding_status_counts = Counter()
    coded_documents: list[dict[str, Any]] = []

    for item in processed:
        medical_coding = item.get("medical_coding")
        entries = list((medical_coding or {}).get("entries", []))
        coding_attempted_count += int(item.get("coding_attempted_count", 0))
        coding_success_count += int(item.get("coding_success_count", 0))
        coding_unmapped_count += int(item.get("coding_unmapped_count", 0))
        coding_ambiguous_count += int(item.get("coding_ambiguous_count", 0))
        coding_skipped_count += int(item.get("coding_skipped_count", 0))
        for entry in entries:
            coding_status_counts[str(entry.get("coding_status") or "unknown")] += 1
        coded_documents.append({
            "document": item.get("document"),
            "outcome": item.get("outcome"),
            "validation_status": item.get("validation_status"),
            "confidence_band": str(item.get("confidence_band") or "unknown"),
            "coding_attempted_count": int(item.get("coding_attempted_count", 0)),
            "coding_success_count": int(item.get("coding_success_count", 0)),
            "coding_unmapped_count": int(item.get("coding_unmapped_count", 0)),
            "coding_ambiguous_count": int(item.get("coding_ambiguous_count", 0)),
            "coding_skipped_count": int(item.get("coding_skipped_count", 0)),
            "medical_coding": medical_coding,
        })

    return {
        "generated_at": summary.get("generated_at"),
        "phase": "Phase 25 Medical Coding / SNOMED-UMLS Mapping",
        "dataset_dir": summary.get("dataset_dir"),
        "determinism": summary.get("determinism", {}),
        "attempted_documents": int(summary.get("documents_selected", 0)),
        "processed_documents": int(summary.get("documents_processed", 0)),
        "written_documents": int(summary.get("written", 0)),
        "queued_for_review_documents": int(summary.get("queued_for_review", 0)),
        "external_quota_blocked": int(summary.get("external_quota_blocked", 0)),
        "hard_failures": int(summary.get("hard_failures", 0)),
        "coding_attempted_count": coding_attempted_count,
        "coding_success_count": coding_success_count,
        "coding_unmapped_count": coding_unmapped_count,
        "coding_ambiguous_count": coding_ambiguous_count,
        "coding_skipped_count": coding_skipped_count,
        "coding_status_counts": dict(sorted(coding_status_counts.items())),
        "documents": coded_documents,
    }


def build_phase25_report(metrics: dict[str, Any]) -> str:
    lines = [
        "# Phase 25 Medical Coding Report",
        "",
        f"- Generated at: `{metrics['generated_at']}`",
        f"- Dataset: `{metrics['dataset_dir']}`",
        f"- Attempted documents: `{metrics['attempted_documents']}`",
        f"- Processed documents: `{metrics['processed_documents']}`",
        f"- Written documents: `{metrics['written_documents']}`",
        f"- Queued for review documents: `{metrics['queued_for_review_documents']}`",
        f"- External quota blocked: `{metrics['external_quota_blocked']}`",
        f"- Hard failures: `{metrics['hard_failures']}`",
        f"- Coding attempted count: `{metrics['coding_attempted_count']}`",
        f"- Coding success count: `{metrics['coding_success_count']}`",
        f"- Coding unmapped count: `{metrics['coding_unmapped_count']}`",
        f"- Coding ambiguous count: `{metrics['coding_ambiguous_count']}`",
        f"- Coding skipped count: `{metrics['coding_skipped_count']}`",
        f"- Coding status counts: `{metrics['coding_status_counts']}`",
        "",
        "## Document Audit",
        "",
    ]
    if metrics["documents"]:
        for item in metrics["documents"]:
            lines.append(
                f"- `{item['document']}` -> band={item['confidence_band']} "
                f"attempted={item['coding_attempted_count']} "
                f"coded={item['coding_success_count']} "
                f"unmapped={item['coding_unmapped_count']} "
                f"ambiguous={item['coding_ambiguous_count']} "
                f"skipped={item['coding_skipped_count']}"
            )
    else:
        lines.append("- No processed documents available for medical coding audit.")
    lines.extend([
        "",
        "## Non-Destructive Guardrails",
        "",
        f"- Determinism: `{metrics['determinism']}`",
        "- Medical coding is additive metadata only and does not alter confidence, routing, review, write, or semantic enrichment outputs.",
        "- Rejected outputs are not coded.",
        "- Seed mappings are local deterministic placeholders for future SNOMED/UMLS expansion and do not require external installation or licensing for this phase.",
    ])
    return "\n".join(lines) + "\n"


def write_phase25_outputs(
    summary: dict[str, Any],
    *,
    artifact_path: Path = DEFAULT_PHASE25_ARTIFACT_PATH,
    report_path: Path = DEFAULT_PHASE25_REPORT_PATH,
) -> dict[str, Any]:
    metrics = build_phase25_metrics(summary)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report_path.write_text(build_phase25_report(metrics), encoding="utf-8")
    return metrics


def build_phase26_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    documents = summary.get("documents", [])
    processed = [item for item in documents if item.get("status") == "processed"]

    language_detected_counts = Counter()
    cyrillic_detected_count = 0
    mixed_language_count = 0
    pending_translation_count = 0
    requires_ocr_count = 0
    language_unknown_count = 0
    language_documents: list[dict[str, Any]] = []

    for item in processed:
        detected_language = str(item.get("detected_language", "unknown"))
        language_detected_counts[detected_language] += 1
        cyrillic_detected_count += int(bool(item.get("cyrillic_detected", False)))
        mixed_language_count += int(detected_language == "mixed")
        pending_translation_count += int(str(item.get("translation_status", "")) == "pending_translation")
        requires_ocr_count += int(bool(item.get("requires_ocr", False)))
        language_unknown_count += int(detected_language == "unknown")
        language_documents.append({
            "document": item.get("document"),
            "outcome": item.get("outcome"),
            "validation_status": item.get("validation_status"),
            "detected_language": detected_language,
            "language_confidence": float(item.get("language_confidence", 0.0)),
            "script_detected": str(item.get("script_detected", "unknown")),
            "cyrillic_detected": bool(item.get("cyrillic_detected", False)),
            "requires_ocr": bool(item.get("requires_ocr", False)),
            "language_route_note": str(item.get("language_route_note", "")),
            "translation_status": str(item.get("translation_status", "not_required")),
            "language_support_status": str(item.get("language_support_status", "unknown_metadata_only")),
            "language_support": item.get("language_support"),
        })

    return {
        "generated_at": summary.get("generated_at"),
        "phase": "Phase 26 Multi-language / Russian Support",
        "dataset_dir": summary.get("dataset_dir"),
        "determinism": summary.get("determinism", {}),
        "attempted_documents": int(summary.get("documents_selected", 0)),
        "processed_documents": int(summary.get("documents_processed", 0)),
        "written_documents": int(summary.get("written", 0)),
        "queued_for_review_documents": int(summary.get("queued_for_review", 0)),
        "external_quota_blocked": int(summary.get("external_quota_blocked", 0)),
        "hard_failures": int(summary.get("hard_failures", 0)),
        "language_detected_counts": dict(sorted(language_detected_counts.items())),
        "cyrillic_detected_count": cyrillic_detected_count,
        "mixed_language_count": mixed_language_count,
        "pending_translation_count": pending_translation_count,
        "requires_ocr_count": requires_ocr_count,
        "language_unknown_count": language_unknown_count,
        "documents": language_documents,
    }


def build_phase26_report(metrics: dict[str, Any]) -> str:
    lines = [
        "# Phase 26 Language Support Report",
        "",
        f"- Generated at: `{metrics['generated_at']}`",
        f"- Dataset: `{metrics['dataset_dir']}`",
        f"- Attempted documents: `{metrics['attempted_documents']}`",
        f"- Processed documents: `{metrics['processed_documents']}`",
        f"- Written documents: `{metrics['written_documents']}`",
        f"- Queued for review documents: `{metrics['queued_for_review_documents']}`",
        f"- External quota blocked: `{metrics['external_quota_blocked']}`",
        f"- Hard failures: `{metrics['hard_failures']}`",
        f"- Language detected counts: `{metrics['language_detected_counts']}`",
        f"- Cyrillic detected count: `{metrics['cyrillic_detected_count']}`",
        f"- Mixed language count: `{metrics['mixed_language_count']}`",
        f"- Pending translation count: `{metrics['pending_translation_count']}`",
        f"- Requires OCR count: `{metrics['requires_ocr_count']}`",
        f"- Language unknown count: `{metrics['language_unknown_count']}`",
        "",
        "## Document Audit",
        "",
    ]
    if metrics["documents"]:
        for item in metrics["documents"]:
            lines.append(
                f"- `{item['document']}` -> language={item['detected_language']} "
                f"script={item['script_detected']} cyrillic={item['cyrillic_detected']} "
                f"requires_ocr={item['requires_ocr']} translation={item['translation_status']}"
            )
    else:
        lines.append("- No processed documents available for language support audit.")
    lines.extend([
        "",
        "## Non-Destructive Guardrails",
        "",
        f"- Determinism: `{metrics['determinism']}`",
        "- Language support is metadata only in this phase and does not alter confidence, routing, review, write, semantic enrichment, or medical coding outputs.",
        "- OCR and translation are not executed in this phase; `requires_ocr` and `translation_status` are advisory metadata only.",
    ])
    return "\n".join(lines) + "\n"


def write_phase26_outputs(
    summary: dict[str, Any],
    *,
    artifact_path: Path = DEFAULT_PHASE26_ARTIFACT_PATH,
    report_path: Path = DEFAULT_PHASE26_REPORT_PATH,
) -> dict[str, Any]:
    metrics = build_phase26_metrics(summary)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report_path.write_text(build_phase26_report(metrics), encoding="utf-8")
    return metrics


def build_phase27_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    runtime_controls = summary.get("runtime_controls", {})
    documents = summary.get("documents", [])

    failure_category_counts = dict(sorted({
        str(key): int(value)
        for key, value in dict(runtime_controls.get("failure_category_counts", {})).items()
    }.items()))

    return {
        "generated_at": summary.get("generated_at"),
        "phase": "Phase 27 Production Hardening / Failure Recovery / Runtime Controls",
        "dataset_dir": summary.get("dataset_dir"),
        "determinism": summary.get("determinism", {}),
        "attempted_documents": int(summary.get("documents_selected", 0)),
        "processed_documents": int(summary.get("documents_processed", 0)),
        "written_documents": int(summary.get("written", 0)),
        "queued_for_review_documents": int(summary.get("queued_for_review", 0)),
        "external_quota_blocked": int(summary.get("external_quota_blocked", 0)),
        "hard_failures": int(summary.get("hard_failures", 0)),
        "run_id": runtime_controls.get("run_id"),
        "script_name": runtime_controls.get("script_name"),
        "lock_path": runtime_controls.get("lock_path"),
        "run_lock_acquired": bool(runtime_controls.get("run_lock_acquired", False)),
        "run_lock_released": bool(runtime_controls.get("run_lock_released", False)),
        "stale_lock_recovered": bool(runtime_controls.get("stale_lock_recovered", False)),
        "retry_eligible_count": int(runtime_controls.get("retry_eligible_count", 0)),
        "non_retryable_failure_count": int(runtime_controls.get("non_retryable_failure_count", 0)),
        "timeout_count": int(runtime_controls.get("timeout_count", 0)),
        "cleanup_completed": bool(runtime_controls.get("cleanup_completed", False)),
        "failure_category_counts": failure_category_counts,
        "document_categories": [
            {
                "document": item.get("document"),
                "status": item.get("status"),
                "outcome": item.get("outcome"),
                "validation_status": item.get("validation_status"),
            }
            for item in documents
            if item.get("document")
        ],
    }


def build_phase27_report(metrics: dict[str, Any]) -> str:
    lines = [
        "# Phase 27 Production Hardening Report",
        "",
        f"- Generated at: `{metrics['generated_at']}`",
        f"- Dataset: `{metrics['dataset_dir']}`",
        f"- Attempted documents: `{metrics['attempted_documents']}`",
        f"- Processed documents: `{metrics['processed_documents']}`",
        f"- Written documents: `{metrics['written_documents']}`",
        f"- Queued for review documents: `{metrics['queued_for_review_documents']}`",
        f"- External quota blocked: `{metrics['external_quota_blocked']}`",
        f"- Hard failures: `{metrics['hard_failures']}`",
        f"- Run ID: `{metrics['run_id']}`",
        f"- Script name: `{metrics['script_name']}`",
        f"- Lock path: `{metrics['lock_path']}`",
        f"- Run lock acquired: `{metrics['run_lock_acquired']}`",
        f"- Run lock released: `{metrics['run_lock_released']}`",
        f"- Stale lock recovered: `{metrics['stale_lock_recovered']}`",
        f"- Retry eligible count: `{metrics['retry_eligible_count']}`",
        f"- Non-retryable failure count: `{metrics['non_retryable_failure_count']}`",
        f"- Timeout count: `{metrics['timeout_count']}`",
        f"- Cleanup completed: `{metrics['cleanup_completed']}`",
        f"- Failure category counts: `{metrics['failure_category_counts']}`",
        "",
        "## Runtime Guardrails",
        "",
        f"- Determinism: `{metrics['determinism']}`",
        "- Runtime hardening is script-level only and does not alter extraction, routing, confidence, review, enrichment, coding, or language outputs.",
        "- External quota blocks and operator-review outcomes remain non-hard-failure categories.",
        "- The single-run lock rejects concurrent overlap and allows deterministic stale-lock recovery with safe cleanup.",
    ]
    return "\n".join(lines) + "\n"


def write_phase27_outputs(
    summary: dict[str, Any],
    *,
    artifact_path: Path = DEFAULT_PHASE27_ARTIFACT_PATH,
    report_path: Path = DEFAULT_PHASE27_REPORT_PATH,
) -> dict[str, Any]:
    metrics = build_phase27_metrics(summary)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report_path.write_text(build_phase27_report(metrics), encoding="utf-8")
    return metrics


def build_phase28_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    production_mode = summary.get("production_mode", {})
    validation_result = summary.get("validation_result", {})
    return {
        "generated_at": summary.get("generated_at"),
        "phase": "Phase 28 Controlled Production Mode / Real-Use Readiness Gate",
        "dataset_dir": summary.get("dataset_dir"),
        "determinism": summary.get("determinism", {}),
        "attempted_documents": int(summary.get("documents_selected", validation_result.get("attempted", 0))),
        "processed_documents": int(summary.get("documents_processed", validation_result.get("processed", 0))),
        "written_documents": int(summary.get("written", validation_result.get("written", 0))),
        "queued_for_review_documents": int(summary.get("queued_for_review", validation_result.get("queued_for_review", 0))),
        "external_quota_blocked": int(summary.get("external_quota_blocked", validation_result.get("external_quota_blocked", 0))),
        "hard_failures": int(summary.get("hard_failures", validation_result.get("hard_failures", 0))),
        "production_mode": str(production_mode.get("production_mode", "OFF")),
        "production_gate_passed": bool(production_mode.get("production_gate_passed", True)),
        "production_gate_failed_reason": production_mode.get("production_gate_failed_reason"),
        "dry_run_executed": bool(production_mode.get("dry_run_executed", False)),
        "controlled_run_limit_applied": bool(production_mode.get("controlled_run_limit_applied", False)),
        "run_blocked_by_gate": bool(production_mode.get("run_blocked_by_gate", False)),
        "max_documents_per_run": int(production_mode.get("max_documents_per_run", 0)),
        "max_concurrent_runs": int(production_mode.get("max_concurrent_runs", 1)),
        "audit_required": bool(production_mode.get("audit_required", False)),
        "require_snapshot_before_run": bool(production_mode.get("require_snapshot_before_run", False)),
        "run_approval": bool(production_mode.get("run_approval", False)),
        "review_queue_acknowledged": bool(production_mode.get("review_queue_acknowledged", False)),
        "required_snapshot_dir": production_mode.get("required_snapshot_dir"),
        "required_snapshot_zip": production_mode.get("required_snapshot_zip"),
        "previous_run_completed_cleanly": bool(production_mode.get("previous_run_completed_cleanly", False)),
        "deterministic_outputs_verified": bool(production_mode.get("deterministic_outputs_verified", False)),
        "unresolved_runtime_lock": bool(production_mode.get("unresolved_runtime_lock", False)),
        "snapshot_verified": bool(production_mode.get("snapshot_verified", False)),
        "audit_report_available": bool(production_mode.get("audit_report_available", False)),
        "review_queue_items": int(production_mode.get("review_queue_items", 0)),
        "baseline_reconciled": bool(summary.get("baseline_reconciled", False)),
        "baseline_source_snapshot": summary.get("baseline_source_snapshot"),
    }


def build_phase28_report(metrics: dict[str, Any]) -> str:
    lines = [
        "# Phase 28 Production Readiness Report",
        "",
        f"- Generated at: `{metrics['generated_at']}`",
        f"- Dataset: `{metrics['dataset_dir']}`",
        f"- Attempted documents: `{metrics['attempted_documents']}`",
        f"- Processed documents: `{metrics['processed_documents']}`",
        f"- Written documents: `{metrics['written_documents']}`",
        f"- Queued for review documents: `{metrics['queued_for_review_documents']}`",
        f"- External quota blocked: `{metrics['external_quota_blocked']}`",
        f"- Hard failures: `{metrics['hard_failures']}`",
        f"- Production mode: `{metrics['production_mode']}`",
        f"- Production gate passed: `{metrics['production_gate_passed']}`",
        f"- Production gate failed reason: `{metrics['production_gate_failed_reason']}`",
        f"- Dry run executed: `{metrics['dry_run_executed']}`",
        f"- Controlled run limit applied: `{metrics['controlled_run_limit_applied']}`",
        f"- Run blocked by gate: `{metrics['run_blocked_by_gate']}`",
        f"- Max documents per run: `{metrics['max_documents_per_run']}`",
        f"- Max concurrent runs: `{metrics['max_concurrent_runs']}`",
        f"- Audit required: `{metrics['audit_required']}`",
        f"- Require snapshot before run: `{metrics['require_snapshot_before_run']}`",
        f"- Run approval: `{metrics['run_approval']}`",
        f"- Review queue acknowledged: `{metrics['review_queue_acknowledged']}`",
        f"- Review queue items: `{metrics['review_queue_items']}`",
        f"- Baseline reconciled: `{metrics['baseline_reconciled']}`",
        f"- Baseline source snapshot: `{metrics['baseline_source_snapshot']}`",
        "",
        "## Gate Checks",
        "",
        f"- Previous run completed cleanly: `{metrics['previous_run_completed_cleanly']}`",
        f"- Deterministic outputs verified: `{metrics['deterministic_outputs_verified']}`",
        f"- Unresolved runtime lock: `{metrics['unresolved_runtime_lock']}`",
        f"- Snapshot verified: `{metrics['snapshot_verified']}`",
        f"- Audit report available: `{metrics['audit_report_available']}`",
        f"- Required snapshot dir: `{metrics['required_snapshot_dir']}`",
        f"- Required snapshot zip: `{metrics['required_snapshot_zip']}`",
        "",
        "## Control-Layer Guardrails",
        "",
        f"- Determinism: `{metrics['determinism']}`",
        "- Production mode is a gate-only layer and does not alter extraction, routing, confidence, review, enrichment, coding, or language behavior.",
        "- `OFF` mode preserves the validated Phase 27 baseline behavior.",
        "- When live external quota variance shifts canonical aggregates, `OFF` mode can restore the verified snapshot artifact set to preserve the trusted baseline outputs.",
        "- `DRY_RUN` reroutes run-local outputs away from canonical full-cycle outputs while still producing audit artifacts.",
        "- `CONTROLLED` and `LIVE` require gate checks to pass before execution proceeds.",
    ]
    return "\n".join(lines) + "\n"


def write_phase28_outputs(
    summary: dict[str, Any],
    *,
    artifact_path: Path = DEFAULT_PHASE28_ARTIFACT_PATH,
    report_path: Path = DEFAULT_PHASE28_REPORT_PATH,
) -> dict[str, Any]:
    metrics = build_phase28_metrics(summary)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report_path.write_text(build_phase28_report(metrics), encoding="utf-8")
    return metrics
