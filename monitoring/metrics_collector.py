from __future__ import annotations

import json
from pathlib import Path

from monitoring.run_record import RunRecord


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUMMARY_PATH = ROOT / "artifacts" / "phase12_real_world_validation" / "phase12_real_world_validation_summary.json"
DEFAULT_AGGREGATE_PATH = ROOT / "reports" / "phase15" / "validation_aggregate.json"
DEFAULT_HISTORY_PATH = ROOT / "reports" / "phase17" / "run_history.jsonl"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_run_record(
    summary: dict,
    aggregate: dict,
    *,
    dataset: str | None = None,
) -> RunRecord:
    outcomes = summary.get("aggregate", {}).get("outcomes", {})
    review_reasons = summary.get("aggregate", {}).get("review_reasons", {})
    documents = summary.get("documents", [])
    determinism = summary.get("determinism", {})
    quota_reasons: dict[str, int] = {}
    external_attempts = 0
    for item in documents:
        retry_visibility = item.get("retry_visibility", {})
        if retry_visibility.get("retry_reason") == "external_quota":
            external_attempts += 1
            reason_key = "external_quota"
            quota_reasons[reason_key] = quota_reasons.get(reason_key, 0) + 1
    generated_at = str(summary.get("generated_at", ""))
    dataset_value = dataset or str(aggregate.get("dataset_dir") or summary.get("dataset_dir") or "")
    dataset_slug = Path(dataset_value).name or "dataset"
    timestamp_slug = generated_at.replace(":", "").replace("-", "").replace("+", "_").replace(".", "")
    duration_sec = round(
        sum(float(item.get("processing_time_ms", 0.0)) for item in documents) / 1000.0,
        3,
    )
    return RunRecord(
        run_id=f"{dataset_slug}-{timestamp_slug}",
        timestamp=generated_at,
        dataset=dataset_value,
        attempted=int(aggregate.get("documents_attempted", summary.get("documents_selected", 0))),
        processed=int(summary.get("documents_processed", aggregate.get("documents_processed", 0))),
        written=int(summary.get("written", 0)),
        written_with_review=int(outcomes.get("written_with_review", 0)),
        external_quota_blocked=int(summary.get("external_quota_blocked", aggregate.get("documents_quota_blocked", 0))),
        hard_failures=int(summary.get("hard_failures", aggregate.get("hard_failures", 0))),
        avg_confidence=float(summary.get("aggregate", {}).get("avg_confidence", aggregate.get("avg_confidence_processed_only", 0.0))),
        route_distribution_requested={
            str(key): int(value)
            for key, value in sorted(aggregate.get("route_distribution_requested", {}).items())
        },
        route_distribution_actual={
            str(key): int(value)
            for key, value in sorted(aggregate.get("route_distribution_actual", {}).items())
        },
        review_counts={
            "clear": int(review_reasons.get("clear", 0)),
            "quarantined": int(review_reasons.get("quarantined", 0)),
        },
        duration_sec=duration_sec,
        determinism={
            "mode": str(determinism.get("mode", "deterministic_path")),
            "seed": determinism.get("seed"),
        },
        quota_behavior={
            "external_attempts": external_attempts,
            "skipped_external_quota": int(summary.get("external_quota_blocked", 0)),
            "reasons": quota_reasons,
        },
    )


def append_run_history(record: RunRecord, history_path: Path = DEFAULT_HISTORY_PATH) -> Path:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
    return history_path


def load_run_history(history_path: Path = DEFAULT_HISTORY_PATH) -> list[dict]:
    if not history_path.exists():
        return []
    return [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def collect_latest_run_metrics(
    *,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
    aggregate_path: Path = DEFAULT_AGGREGATE_PATH,
    history_path: Path = DEFAULT_HISTORY_PATH,
) -> RunRecord:
    summary = load_json(summary_path)
    aggregate = load_json(aggregate_path)
    record = build_run_record(summary, aggregate)
    append_run_history(record, history_path=history_path)
    return record
