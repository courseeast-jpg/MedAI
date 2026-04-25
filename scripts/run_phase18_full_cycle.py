from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monitoring.run_comparator import write_stability_report

PHASE18_REPORT_DIR = ROOT / "reports" / "phase18"
PHASE11_AUDIT_PATH = ROOT / "artifacts" / "phase11_integration" / "phase11_integration_audit.json"
PHASE12_SUMMARY_PATH = ROOT / "artifacts" / "phase12_real_world_validation" / "phase12_real_world_validation_summary.json"
PHASE21_METRICS_PATH = ROOT / "artifacts" / "phase21" / "observability_metrics.json"
PHASE21_REPORT_PATH = ROOT / "reports" / "phase21" / "observability_report.md"
PHASE22_METRICS_PATH = ROOT / "artifacts" / "phase22" / "confidence_calibration.json"
PHASE22_REPORT_PATH = ROOT / "reports" / "phase22" / "accuracy_calibration_report.md"
PHASE23_METRICS_PATH = ROOT / "artifacts" / "phase23" / "routing_efficiency.json"
PHASE23_REPORT_PATH = ROOT / "reports" / "phase23" / "routing_efficiency_report.md"
PHASE24_METRICS_PATH = ROOT / "artifacts" / "phase24" / "semantic_enrichment.json"
PHASE24_REPORT_PATH = ROOT / "reports" / "phase24" / "semantic_enrichment_report.md"
PHASE25_METRICS_PATH = ROOT / "artifacts" / "phase25" / "medical_coding.json"
PHASE25_REPORT_PATH = ROOT / "reports" / "phase25" / "medical_coding_report.md"
PHASE17_DASHBOARD_PATH = ROOT / "reports" / "phase17" / "dashboard_latest.md"

PHASE18_STEPS: list[tuple[str, list[str]]] = [
    ("tests", [sys.executable, "-m", "pytest", "tests"]),
    ("phase11_audit", [sys.executable, "scripts\\run_phase11_integration_audit.py"]),
    (
        "validation",
        [
            sys.executable,
            "scripts\\run_phase12_real_world_validation.py",
            "--dataset-dir",
            "test_data\\final_batch_50",
            "--quota-safe",
        ],
    ),
    ("dashboard_latest", [sys.executable, "scripts\\run_phase17_dashboard.py", "--latest"]),
    ("dashboard_export", [sys.executable, "scripts\\run_phase17_dashboard.py", "--export"]),
]

PYTEST_SUMMARY_RE = re.compile(r"=+\s+(\d+)\s+passed(?:,.*)?\s+=+")


def run_command(command: list[str]) -> dict:
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def summarize_pytest(stdout: str) -> str:
    match = PYTEST_SUMMARY_RE.search(stdout)
    if match:
        return f"{match.group(1)} passed"
    return "unknown"


def git_commit_hash() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def git_status_state() -> str:
    completed = subprocess.run(
        ["git", "status", "--short"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return "unknown"
    return "clean" if not completed.stdout.strip() else "dirty"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_summary(*, commands: list[dict], started_at: datetime, ended_at: datetime) -> dict:
    phase11 = load_json(PHASE11_AUDIT_PATH) if PHASE11_AUDIT_PATH.exists() else {}
    phase12 = load_json(PHASE12_SUMMARY_PATH) if PHASE12_SUMMARY_PATH.exists() else {}
    phase21 = load_json(PHASE21_METRICS_PATH) if PHASE21_METRICS_PATH.exists() else {}
    phase22 = load_json(PHASE22_METRICS_PATH) if PHASE22_METRICS_PATH.exists() else {}
    phase23 = load_json(PHASE23_METRICS_PATH) if PHASE23_METRICS_PATH.exists() else {}
    phase24 = load_json(PHASE24_METRICS_PATH) if PHASE24_METRICS_PATH.exists() else {}
    phase25 = load_json(PHASE25_METRICS_PATH) if PHASE25_METRICS_PATH.exists() else {}
    pytest_step = next((item for item in commands if item["name"] == "tests"), None)
    failed_step = next((item["name"] for item in commands if item["returncode"] != 0), None)

    return {
        "generated_at": ended_at.isoformat(),
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "duration_seconds": round((ended_at - started_at).total_seconds(), 3),
        "commit_hash": git_commit_hash(),
        "git_status": git_status_state(),
        "steps": [
            {
                "name": item["name"],
                "command": item["command"],
                "returncode": item["returncode"],
            }
            for item in commands
        ],
        "success": failed_step is None,
        "failed_step": failed_step,
        "test_result": summarize_pytest(pytest_step["stdout"]) if pytest_step else "unknown",
        "phase11_audit_result": "passed" if phase11.get("merge_recommended") else "failed",
        "validation_result": {
            "attempted": int(phase12.get("documents_selected", 0)),
            "processed": int(phase12.get("documents_processed", 0)),
            "written": int(phase12.get("written", 0)),
            "queued_for_review": int(phase12.get("queued_for_review", 0)),
            "external_quota_blocked": int(phase12.get("external_quota_blocked", 0)),
            "hard_failures": int(phase12.get("hard_failures", 0)),
            "avg_confidence": float(phase12.get("aggregate", {}).get("avg_confidence", 0.0)),
            "review_queue_items": int(phase12.get("review_queue", {}).get("items", 0)),
            "review_queue_path": phase12.get("review_queue", {}).get("path"),
        },
        "observability_result": {
            "metrics_path": str(PHASE21_METRICS_PATH),
            "report_path": str(PHASE21_REPORT_PATH),
            "route_mismatch_count": int(phase21.get("route_mismatch_count", 0)),
            "low_confidence_count": int(phase21.get("low_confidence_count", 0)),
            "quota_safe_block_count": int(phase21.get("quota_safe_block_count", 0)),
            "extractor_route_counts": phase21.get("extractor_route_counts", {}),
            "extractor_actual_counts": phase21.get("extractor_actual_counts", {}),
            "per_stage_duration_ms": phase21.get("per_stage_duration_ms", {}),
        },
        "calibration_result": {
            "metrics_path": str(PHASE22_METRICS_PATH),
            "report_path": str(PHASE22_REPORT_PATH),
            "average_raw_confidence": float(phase22.get("average_raw_confidence", 0.0)),
            "average_calibrated_confidence": float(phase22.get("average_calibrated_confidence", 0.0)),
            "confidence_band_counts": phase22.get("confidence_band_counts", {}),
            "review_recommendation_counts": phase22.get("review_recommendation_counts", {}),
            "route_mismatch_count": int(phase22.get("route_mismatch_count", 0)),
        },
        "routing_efficiency_result": {
            "metrics_path": str(PHASE23_METRICS_PATH),
            "report_path": str(PHASE23_REPORT_PATH),
            "intended_route_counts": phase23.get("intended_route_counts", {}),
            "actual_route_counts": phase23.get("actual_route_counts", {}),
            "route_mismatch_count": int(phase23.get("route_mismatch_count", 0)),
            "quota_block_avoided_count": int(phase23.get("quota_block_avoided_count", 0)),
            "total_estimated_cost_units": float(phase23.get("total_estimated_cost_units", 0.0)),
            "total_saved_cost_units": float(phase23.get("total_saved_cost_units", 0.0)),
        },
        "semantic_enrichment_result": {
            "metrics_path": str(PHASE24_METRICS_PATH),
            "report_path": str(PHASE24_REPORT_PATH),
            "enrichment_applied_count": int(phase24.get("enrichment_applied_count", 0)),
            "negation_detected_count": int(phase24.get("negation_detected_count", 0)),
            "temporal_detected_count": int(phase24.get("temporal_detected_count", 0)),
            "relationships_detected_count": int(phase24.get("relationships_detected_count", 0)),
        },
        "medical_coding_result": {
            "metrics_path": str(PHASE25_METRICS_PATH),
            "report_path": str(PHASE25_REPORT_PATH),
            "coding_attempted_count": int(phase25.get("coding_attempted_count", 0)),
            "coding_success_count": int(phase25.get("coding_success_count", 0)),
            "coding_unmapped_count": int(phase25.get("coding_unmapped_count", 0)),
            "coding_ambiguous_count": int(phase25.get("coding_ambiguous_count", 0)),
            "coding_skipped_count": int(phase25.get("coding_skipped_count", 0)),
            "coding_status_counts": phase25.get("coding_status_counts", {}),
        },
        "dashboard_export_path": str(PHASE17_DASHBOARD_PATH),
        "stability_report_path": str(ROOT / "reports" / "phase19" / "stability_report.md"),
    }


def write_summary_reports(summary: dict, report_dir: Path = PHASE18_REPORT_DIR) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "full_cycle_summary.json"
    md_path = report_dir / "full_cycle_summary.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    validation = summary["validation_result"]
    observability = summary["observability_result"]
    calibration = summary["calibration_result"]
    routing_efficiency = summary["routing_efficiency_result"]
    semantic_enrichment = summary["semantic_enrichment_result"]
    medical_coding = summary["medical_coding_result"]
    lines = [
        "# Phase 18 Full Cycle Summary",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Commit hash: `{summary['commit_hash']}`",
        f"- Git status: `{summary['git_status']}`",
        f"- Success: `{summary['success']}`",
        f"- Failed step: `{summary['failed_step']}`",
        f"- Test result: `{summary['test_result']}`",
        f"- Phase 11 audit result: `{summary['phase11_audit_result']}`",
        f"- Validation attempted: `{validation['attempted']}`",
        f"- Validation processed: `{validation['processed']}`",
        f"- Validation written: `{validation['written']}`",
        f"- Validation queued_for_review: `{validation['queued_for_review']}`",
        f"- Validation external_quota_blocked: `{validation['external_quota_blocked']}`",
        f"- Validation hard_failures: `{validation['hard_failures']}`",
        f"- Validation avg_confidence: `{validation['avg_confidence']}`",
        f"- Validation review_queue_items: `{validation['review_queue_items']}`",
        f"- Validation review_queue_path: `{validation['review_queue_path']}`",
        f"- Observability route_mismatch_count: `{observability['route_mismatch_count']}`",
        f"- Observability low_confidence_count: `{observability['low_confidence_count']}`",
        f"- Observability quota_safe_block_count: `{observability['quota_safe_block_count']}`",
        f"- Observability metrics_path: `{observability['metrics_path']}`",
        f"- Observability report_path: `{observability['report_path']}`",
        f"- Calibration average_raw_confidence: `{calibration['average_raw_confidence']}`",
        f"- Calibration average_calibrated_confidence: `{calibration['average_calibrated_confidence']}`",
        f"- Calibration confidence_band_counts: `{calibration['confidence_band_counts']}`",
        f"- Calibration review_recommendation_counts: `{calibration['review_recommendation_counts']}`",
        f"- Calibration route_mismatch_count: `{calibration['route_mismatch_count']}`",
        f"- Calibration metrics_path: `{calibration['metrics_path']}`",
        f"- Calibration report_path: `{calibration['report_path']}`",
        f"- Routing efficiency intended_route_counts: `{routing_efficiency['intended_route_counts']}`",
        f"- Routing efficiency actual_route_counts: `{routing_efficiency['actual_route_counts']}`",
        f"- Routing efficiency route_mismatch_count: `{routing_efficiency['route_mismatch_count']}`",
        f"- Routing efficiency quota_block_avoided_count: `{routing_efficiency['quota_block_avoided_count']}`",
        f"- Routing efficiency total_estimated_cost_units: `{routing_efficiency['total_estimated_cost_units']}`",
        f"- Routing efficiency total_saved_cost_units: `{routing_efficiency['total_saved_cost_units']}`",
        f"- Routing efficiency metrics_path: `{routing_efficiency['metrics_path']}`",
        f"- Routing efficiency report_path: `{routing_efficiency['report_path']}`",
        f"- Semantic enrichment applied_count: `{semantic_enrichment['enrichment_applied_count']}`",
        f"- Semantic enrichment negation_detected_count: `{semantic_enrichment['negation_detected_count']}`",
        f"- Semantic enrichment temporal_detected_count: `{semantic_enrichment['temporal_detected_count']}`",
        f"- Semantic enrichment relationships_detected_count: `{semantic_enrichment['relationships_detected_count']}`",
        f"- Semantic enrichment metrics_path: `{semantic_enrichment['metrics_path']}`",
        f"- Semantic enrichment report_path: `{semantic_enrichment['report_path']}`",
        f"- Medical coding attempted_count: `{medical_coding['coding_attempted_count']}`",
        f"- Medical coding success_count: `{medical_coding['coding_success_count']}`",
        f"- Medical coding unmapped_count: `{medical_coding['coding_unmapped_count']}`",
        f"- Medical coding ambiguous_count: `{medical_coding['coding_ambiguous_count']}`",
        f"- Medical coding skipped_count: `{medical_coding['coding_skipped_count']}`",
        f"- Medical coding status_counts: `{medical_coding['coding_status_counts']}`",
        f"- Medical coding metrics_path: `{medical_coding['metrics_path']}`",
        f"- Medical coding report_path: `{medical_coding['report_path']}`",
        f"- Dashboard export path: `{summary['dashboard_export_path']}`",
        f"- Stability report path: `{summary['stability_report_path']}`",
        f"- Duration seconds: `{summary['duration_seconds']}`",
        "",
        "## Steps",
        "",
    ]
    lines.extend(
        f"- `{item['name']}` -> returncode={item['returncode']} command={' '.join(item['command'])}"
        for item in summary["steps"]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def execute_steps(
    steps: list[tuple[str, list[str]]] = PHASE18_STEPS,
    *,
    runner=run_command,
) -> list[dict]:
    results: list[dict] = []
    for name, command in steps:
        result = runner(command)
        results.append({
            "name": name,
            **result,
        })
        if result["returncode"] != 0:
            break
    return results


def main() -> int:
    started_at = datetime.now(UTC)
    results = execute_steps()
    ended_at = datetime.now(UTC)
    summary = build_summary(commands=results, started_at=started_at, ended_at=ended_at)
    write_summary_reports(summary)
    write_stability_report()
    print(json.dumps(summary, indent=2))
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
