from __future__ import annotations

from pathlib import Path

import yaml

from monitoring.metrics_collector import DEFAULT_HISTORY_PATH, load_run_history


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOLERANCE_PATH = ROOT / "config" / "phase19_stability.yaml"
DEFAULT_REPORT_PATH = ROOT / "reports" / "phase19" / "stability_report.md"


def load_tolerances(config_path: Path = DEFAULT_TOLERANCE_PATH) -> dict[str, float]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return {
        "processed_delta_max": float(data.get("processed_delta_max", 2)),
        "written_delta_max": float(data.get("written_delta_max", 2)),
        "queued_delta_max": float(data.get("queued_delta_max", 2)),
        "confidence_delta_max": float(data.get("confidence_delta_max", 0.05)),
    }


def queued_documents(record: dict) -> int:
    return int(record["attempted"]) - int(record["written"]) - int(record["external_quota_blocked"]) - int(record["hard_failures"])


def compare_runs(current: dict, previous: dict, tolerances: dict[str, float]) -> dict:
    deltas = {
        "processed": int(current["processed"]) - int(previous["processed"]),
        "written": int(current["written"]) - int(previous["written"]),
        "queued_for_review": queued_documents(current) - queued_documents(previous),
        "quota_blocked": int(current["external_quota_blocked"]) - int(previous["external_quota_blocked"]),
        "avg_confidence": round(float(current["avg_confidence"]) - float(previous["avg_confidence"]), 3),
    }
    exceeded = {
        "processed": abs(deltas["processed"]) > tolerances["processed_delta_max"],
        "written": abs(deltas["written"]) > tolerances["written_delta_max"],
        "queued_for_review": abs(deltas["queued_for_review"]) > tolerances["queued_delta_max"],
        "avg_confidence": abs(deltas["avg_confidence"]) > tolerances["confidence_delta_max"],
    }
    return {
        "current_run_id": current["run_id"],
        "previous_run_id": previous["run_id"],
        "current": current,
        "previous": previous,
        "deltas": deltas,
        "tolerances": tolerances,
        "exceeded": exceeded,
        "status": "UNSTABLE" if any(exceeded.values()) else "STABLE",
    }


def compare_last_runs(
    *,
    history_path: Path = DEFAULT_HISTORY_PATH,
    config_path: Path = DEFAULT_TOLERANCE_PATH,
    limit: int = 2,
) -> dict:
    history = load_run_history(history_path)
    if len(history) < 2:
        return {
            "status": "INSUFFICIENT_HISTORY",
            "history": history[-limit:],
            "comparisons": [],
            "tolerances": load_tolerances(config_path),
        }

    visible = history[-max(limit, 2):]
    tolerances = load_tolerances(config_path)
    comparisons = [
        compare_runs(visible[index], visible[index - 1], tolerances)
        for index in range(1, len(visible))
    ]
    return {
        "status": comparisons[-1]["status"],
        "history": visible,
        "comparisons": comparisons,
        "tolerances": tolerances,
    }


def explain_variance(comparison: dict) -> list[str]:
    explanations: list[str] = []
    deltas = comparison["deltas"]
    if deltas["quota_blocked"] != 0:
        explanations.append(
            f"Quota-blocked documents changed by {deltas['quota_blocked']}, which can shift processed totals without any pipeline behavior change."
        )
    if deltas["queued_for_review"] != 0:
        explanations.append(
            f"Derived queued document count changed by {deltas['queued_for_review']}; inspect review counts and quota behavior for this run."
        )
    if deltas["avg_confidence"] != 0:
        explanations.append(
            f"Average confidence changed by {deltas['avg_confidence']:+.3f}; compare document mix and quota-skipped documents."
        )
    if not explanations:
        explanations.append("No material variance detected between the compared runs.")
    return explanations


def write_stability_report(
    *,
    history_path: Path = DEFAULT_HISTORY_PATH,
    config_path: Path = DEFAULT_TOLERANCE_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    limit: int = 3,
) -> Path:
    comparison_bundle = compare_last_runs(history_path=history_path, config_path=config_path, limit=limit)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 19 Stability Report",
        "",
        f"- Overall status: `{comparison_bundle['status']}`",
        f"- Tolerances: `{comparison_bundle['tolerances']}`",
        "",
        "## Recent Runs",
        "",
    ]
    for item in comparison_bundle["history"]:
        lines.append(
            f"- `{item['run_id']}` -> processed={item['processed']} written={item['written']} queued={queued_documents(item)} quota={item['external_quota_blocked']} avg_conf={float(item['avg_confidence']):.3f}"
        )
    lines.extend(["", "## Comparisons", ""])
    if comparison_bundle["comparisons"]:
        for item in comparison_bundle["comparisons"]:
            lines.append(
                f"- `{item['previous_run_id']}` -> `{item['current_run_id']}` status={item['status']} deltas={item['deltas']}"
            )
            for explanation in explain_variance(item):
                lines.append(f"  explanation: {explanation}")
    else:
        lines.append("- Not enough run history yet to compare multiple runs.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
