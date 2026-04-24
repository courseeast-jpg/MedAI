from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HISTORY_PATH = ROOT / "reports" / "phase17" / "run_history.jsonl"
DEFAULT_EXPORT_PATH = ROOT / "reports" / "phase17" / "dashboard_latest.md"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monitoring.metrics_collector import DEFAULT_AGGREGATE_PATH, DEFAULT_SUMMARY_PATH, load_run_history


def load_latest_record(history_path: Path) -> dict | None:
    history = load_run_history(history_path)
    return history[-1] if history else None


def compute_delta(current: dict, previous: dict | None) -> dict[str, str]:
    if previous is None:
        return {"written": "n/a", "queued": "n/a", "confidence": "n/a", "duration": "n/a"}

    def delta_str(field: str, *, fmt: str = "{:+d}") -> str:
        current_value = current[field]
        previous_value = previous[field]
        if isinstance(current_value, float) or isinstance(previous_value, float):
            return f"{float(current_value) - float(previous_value):+.3f}"
        return fmt.format(int(current_value) - int(previous_value))

    queued_current = int(current["attempted"]) - int(current["written"]) - int(current["external_quota_blocked"]) - int(current["hard_failures"])
    queued_previous = int(previous["attempted"]) - int(previous["written"]) - int(previous["external_quota_blocked"]) - int(previous["hard_failures"])
    return {
        "written": delta_str("written"),
        "queued": f"{queued_current - queued_previous:+d}",
        "confidence": delta_str("avg_confidence"),
        "duration": delta_str("duration_sec"),
    }


def render_table(rows: list[dict]) -> str:
    headers = [
        "run_id",
        "attempted",
        "processed",
        "written",
        "w_review",
        "quota",
        "fail",
        "avg_conf",
        "dur_s",
        "d_written",
        "d_queue",
    ]
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row.get(header, ""))))

    def format_row(row: dict) -> str:
        return " | ".join(str(row.get(header, "")).ljust(widths[header]) for header in headers)

    separator = "-+-".join("-" * widths[header] for header in headers)
    lines = [format_row({header: header for header in headers}), separator]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)


def render_latest_summary(latest: dict, previous: dict | None) -> str:
    delta = compute_delta(latest, previous)
    queued_documents = latest["attempted"] - latest["written"] - latest["external_quota_blocked"] - latest["hard_failures"]
    row = {
        "run_id": latest["run_id"],
        "attempted": latest["attempted"],
        "processed": latest["processed"],
        "written": latest["written"],
        "w_review": latest["written_with_review"],
        "quota": latest["external_quota_blocked"],
        "fail": latest["hard_failures"],
        "avg_conf": f"{float(latest['avg_confidence']):.3f}",
        "dur_s": f"{float(latest['duration_sec']):.3f}",
        "d_written": delta["written"],
        "d_queue": delta["queued"],
    }
    details = [
        "Phase 17 Latest Run",
        "",
        render_table([row]),
        "",
        f"dataset: {latest['dataset']}",
        f"requested routes: {latest['route_distribution_requested']}",
        f"actual routes: {latest['route_distribution_actual']}",
        f"review counts: {latest['review_counts']}",
        f"derived queued documents: {queued_documents}",
        f"delta avg_confidence: {delta['confidence']}",
        f"delta duration_sec: {delta['duration']}",
    ]
    return "\n".join(details)


def render_history_summary(history: list[dict], limit: int) -> str:
    rows = []
    visible = history[-limit:]
    for index, item in enumerate(visible):
        previous = visible[index - 1] if index > 0 else None
        delta = compute_delta(item, previous)
        rows.append({
            "run_id": item["run_id"],
            "attempted": item["attempted"],
            "processed": item["processed"],
            "written": item["written"],
            "w_review": item["written_with_review"],
            "quota": item["external_quota_blocked"],
            "fail": item["hard_failures"],
            "avg_conf": f"{float(item['avg_confidence']):.3f}",
            "dur_s": f"{float(item['duration_sec']):.3f}",
            "d_written": delta["written"],
            "d_queue": delta["queued"],
        })
    return "\n".join(["Phase 17 Run History", "", render_table(rows)])


def export_latest_markdown(latest: dict, previous: dict | None, export_path: Path) -> Path:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    delta = compute_delta(latest, previous)
    queued_documents = latest["attempted"] - latest["written"] - latest["external_quota_blocked"] - latest["hard_failures"]
    lines = [
        "# Phase 17 Dashboard",
        "",
        f"- run_id: `{latest['run_id']}`",
        f"- timestamp: `{latest['timestamp']}`",
        f"- dataset: `{latest['dataset']}`",
        f"- attempted: `{latest['attempted']}`",
        f"- processed: `{latest['processed']}`",
        f"- written: `{latest['written']}`",
        f"- written_with_review: `{latest['written_with_review']}`",
        f"- external_quota_blocked: `{latest['external_quota_blocked']}`",
        f"- hard_failures: `{latest['hard_failures']}`",
        f"- avg_confidence: `{float(latest['avg_confidence']):.3f}`",
        f"- duration_sec: `{float(latest['duration_sec']):.3f}`",
        f"- derived_queued_documents: `{queued_documents}`",
        f"- delta_written_vs_previous: `{delta['written']}`",
        f"- delta_queued_vs_previous: `{delta['queued']}`",
        f"- route_distribution_requested: `{latest['route_distribution_requested']}`",
        f"- route_distribution_actual: `{latest['route_distribution_actual']}`",
        f"- review_counts: `{latest['review_counts']}`",
        "",
        "## Inputs To Outputs",
        "",
        f"- Source artifacts: `{DEFAULT_SUMMARY_PATH}` and `{DEFAULT_AGGREGATE_PATH}`",
        "- Input: latest validation summary + aggregate",
        "- Decision layer: read-only metrics collector",
        "- Output: appended run history + dashboard view",
    ]
    export_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return export_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--history", type=int, default=0)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--history-path", default=str(DEFAULT_HISTORY_PATH))
    args = parser.parse_args()

    history_path = Path(args.history_path)
    history = load_run_history(history_path)
    if not history:
        raise SystemExit(f"No run history found at {history_path}")

    latest = history[-1]
    previous = history[-2] if len(history) > 1 else None

    if args.history:
        print(render_history_summary(history, args.history))
    else:
        print(render_latest_summary(latest, previous))

    if args.export or args.latest:
        export_latest_markdown(latest, previous, DEFAULT_EXPORT_PATH)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
