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

        extractor_route_counts[route_value] += 1
        extractor_actual_counts[actual_value] += 1
        if requested_value != "unknown" and requested_value != actual_value:
            route_mismatch_count += 1
        if float(item.get("confidence", 0.0)) < EXTRACTION_ACCEPT_THRESHOLD:
            low_confidence_count += 1

    review_queue_categories = Counter(
        str(item.get("queue_category", "unknown"))
        for item in review_queue_items
    )

    quota_safe_block_count = sum(
        1 for item in documents
        if item.get("status") == "external_quota_blocked"
    )

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
