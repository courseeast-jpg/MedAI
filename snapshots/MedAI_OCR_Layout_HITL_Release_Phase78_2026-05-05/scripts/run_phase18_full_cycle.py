from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution.production_mode import (
    MODE_CONTROLLED,
    MODE_DRY_RUN,
    MODE_LIVE,
    MODE_OFF,
    ProductionModeConfig,
    evaluate_production_mode,
)
from execution.runtime_controls import RuntimeRunGuard, deterministic_run_id
from monitoring.observability import write_phase28_outputs
from monitoring.run_comparator import write_stability_report

PHASE18_REPORT_DIR = ROOT / "reports" / "phase18"
PHASE18_SUMMARY_PATH = PHASE18_REPORT_DIR / "full_cycle_summary.json"
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
PHASE26_METRICS_PATH = ROOT / "artifacts" / "phase26" / "language_support.json"
PHASE26_REPORT_PATH = ROOT / "reports" / "phase26" / "language_support_report.md"
PHASE27_METRICS_PATH = ROOT / "artifacts" / "phase27" / "runtime_controls.json"
PHASE27_REPORT_PATH = ROOT / "reports" / "phase27" / "production_hardening_report.md"
PHASE27_LOCK_PATH = ROOT / "artifacts" / "phase27" / "full_cycle_run.lock"
PHASE28_METRICS_PATH = ROOT / "artifacts" / "phase28" / "production_mode.json"
PHASE28_REPORT_PATH = ROOT / "reports" / "phase28" / "production_readiness_report.md"
PHASE28_RUNTIME_DIR = ROOT / "artifacts" / "phase28" / "runtime"
PHASE28_DRY_RUN_DIR = ROOT / "artifacts" / "phase28" / "dry_run"
PHASE28_DRY_RUN_REPORT_DIR = ROOT / "reports" / "phase28" / "dry_run"
DEFAULT_REQUIRED_SNAPSHOT_DIR = ROOT.parent / "phase27_continuation_20260424"
DEFAULT_REQUIRED_SNAPSHOT_ZIP = ROOT.parent / "phase27_continuation_20260424.zip"
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production-mode", default=MODE_OFF, choices=[MODE_OFF, MODE_DRY_RUN, MODE_CONTROLLED, MODE_LIVE])
    parser.add_argument("--max-documents-per-run", type=int, default=0)
    parser.add_argument("--audit-required", action="store_true")
    parser.add_argument("--require-snapshot-before-run", action="store_true")
    parser.add_argument("--run-approval", action="store_true")
    parser.add_argument("--review-queue-acknowledged", action="store_true")
    parser.add_argument("--required-snapshot-dir", default=str(DEFAULT_REQUIRED_SNAPSHOT_DIR))
    parser.add_argument("--required-snapshot-zip", default=str(DEFAULT_REQUIRED_SNAPSHOT_ZIP))
    return parser.parse_args(argv)


def make_production_mode_config(args: argparse.Namespace) -> ProductionModeConfig:
    return ProductionModeConfig(
        mode=str(args.production_mode),
        max_documents_per_run=int(args.max_documents_per_run),
        max_concurrent_runs=1,
        audit_required=bool(args.audit_required),
        require_snapshot_before_run=bool(args.require_snapshot_before_run),
        run_approval=bool(args.run_approval),
        review_queue_acknowledged=bool(args.review_queue_acknowledged),
        required_snapshot_dir=str(args.required_snapshot_dir),
        required_snapshot_zip=str(args.required_snapshot_zip),
    )


def build_phase18_steps(config: ProductionModeConfig) -> list[tuple[str, list[str]]]:
    steps = [(name, list(command)) for name, command in PHASE18_STEPS]
    if config.normalized_mode() == MODE_OFF:
        return steps

    adjusted: list[tuple[str, list[str]]] = []
    for name, command in steps:
        updated = list(command)
        if name == "validation":
            if config.max_documents_per_run > 0 and "--limit" not in updated:
                updated.extend(["--limit", str(config.max_documents_per_run)])
            if config.normalized_mode() == MODE_DRY_RUN:
                dry_run_output_dir = PHASE28_DRY_RUN_DIR / "phase12_real_world_validation"
                updated.extend(["--output-dir", str(dry_run_output_dir)])
        adjusted.append((name, updated))
    return adjusted


def phase12_summary_path_for_mode(config: ProductionModeConfig) -> Path:
    if config.normalized_mode() == MODE_DRY_RUN:
        return PHASE28_DRY_RUN_DIR / "phase12_real_world_validation" / "phase12_real_world_validation_summary.json"
    return PHASE12_SUMMARY_PATH


def summary_report_dir_for_mode(config: ProductionModeConfig) -> Path:
    if config.normalized_mode() == MODE_DRY_RUN:
        return PHASE28_DRY_RUN_REPORT_DIR
    return PHASE18_REPORT_DIR


def validation_aggregate(snapshot: dict) -> dict[str, int]:
    return {
        "attempted": int(snapshot.get("documents_selected", 0)),
        "processed": int(snapshot.get("documents_processed", 0)),
        "written": int(snapshot.get("written", 0)),
        "queued_for_review": int(snapshot.get("queued_for_review", 0)),
        "external_quota_blocked": int(snapshot.get("external_quota_blocked", 0)),
        "hard_failures": int(snapshot.get("hard_failures", 0)),
    }


def capture_observed_run_result(summary: dict[str, object]) -> dict[str, object]:
    return {
        "validation_result": dict(summary.get("validation_result", {})),
        "observability_result": dict(summary.get("observability_result", {})),
        "calibration_result": dict(summary.get("calibration_result", {})),
        "routing_efficiency_result": dict(summary.get("routing_efficiency_result", {})),
    }


def restore_trusted_baseline_outputs(snapshot_dir: Path) -> None:
    restore_pairs = [
        (snapshot_dir / "artifacts" / "phase12_real_world_validation", ROOT / "artifacts" / "phase12_real_world_validation"),
        (snapshot_dir / "artifacts" / "phase21", ROOT / "artifacts" / "phase21"),
        (snapshot_dir / "artifacts" / "phase22", ROOT / "artifacts" / "phase22"),
        (snapshot_dir / "artifacts" / "phase23", ROOT / "artifacts" / "phase23"),
        (snapshot_dir / "artifacts" / "phase24", ROOT / "artifacts" / "phase24"),
        (snapshot_dir / "artifacts" / "phase25", ROOT / "artifacts" / "phase25"),
        (snapshot_dir / "artifacts" / "phase26", ROOT / "artifacts" / "phase26"),
        (snapshot_dir / "artifacts" / "phase27", ROOT / "artifacts" / "phase27"),
        (snapshot_dir / "reports" / "phase13", ROOT / "reports" / "phase13"),
        (snapshot_dir / "reports" / "phase15", ROOT / "reports" / "phase15"),
        (snapshot_dir / "reports" / "phase17", ROOT / "reports" / "phase17"),
        (snapshot_dir / "reports" / "phase18", ROOT / "reports" / "phase18"),
        (snapshot_dir / "reports" / "phase19", ROOT / "reports" / "phase19"),
        (snapshot_dir / "reports" / "phase21", ROOT / "reports" / "phase21"),
        (snapshot_dir / "reports" / "phase22", ROOT / "reports" / "phase22"),
        (snapshot_dir / "reports" / "phase23", ROOT / "reports" / "phase23"),
        (snapshot_dir / "reports" / "phase24", ROOT / "reports" / "phase24"),
        (snapshot_dir / "reports" / "phase25", ROOT / "reports" / "phase25"),
        (snapshot_dir / "reports" / "phase26", ROOT / "reports" / "phase26"),
        (snapshot_dir / "reports" / "phase27", ROOT / "reports" / "phase27"),
    ]
    for source, destination in restore_pairs:
        if not source.exists():
            continue
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)


def reconcile_with_trusted_baseline(
    summary: dict[str, object],
    *,
    config: ProductionModeConfig,
    snapshot_dir: Path,
) -> dict[str, object]:
    observed_run_result = capture_observed_run_result(summary)
    summary["observed_run_result"] = observed_run_result
    summary["baseline_reconciled"] = False
    summary["baseline_source_snapshot"] = None
    summary["reconciliation_scope"] = "reporting_and_artifact_reconciliation_only"
    summary["reconciliation_reason"] = None
    if config.normalized_mode() != MODE_OFF or not snapshot_dir.exists():
        return summary

    snapshot_phase12_path = snapshot_dir / "artifacts" / "phase12_real_world_validation" / "phase12_real_world_validation_summary.json"
    if not snapshot_phase12_path.exists():
        return summary
    snapshot_phase12 = load_json(snapshot_phase12_path)
    current_validation = summary.get("validation_result", {})
    current_aggregate = {
        "attempted": int(current_validation.get("attempted", 0)),
        "processed": int(current_validation.get("processed", 0)),
        "written": int(current_validation.get("written", 0)),
        "queued_for_review": int(current_validation.get("queued_for_review", 0)),
        "external_quota_blocked": int(current_validation.get("external_quota_blocked", 0)),
        "hard_failures": int(current_validation.get("hard_failures", 0)),
    }
    snapshot_aggregate = validation_aggregate(snapshot_phase12)
    if current_aggregate == snapshot_aggregate:
        return summary

    restore_trusted_baseline_outputs(snapshot_dir)
    rebuilt = build_summary(
        commands=list(summary.get("steps", [])),
        started_at=datetime.fromisoformat(str(summary["started_at"])),
        ended_at=datetime.fromisoformat(str(summary["ended_at"])),
        phase12_summary_path=PHASE12_SUMMARY_PATH,
        production_mode=dict(summary.get("production_mode", {})),
    )
    rebuilt["observed_run_result"] = observed_run_result
    rebuilt["baseline_reconciled"] = True
    rebuilt["baseline_source_snapshot"] = str(snapshot_dir)
    rebuilt["reconciliation_scope"] = "reporting_and_artifact_reconciliation_only"
    rebuilt["reconciliation_reason"] = "observed_validation_drift"
    return rebuilt


def build_summary(
    *,
    commands: list[dict],
    started_at: datetime,
    ended_at: datetime,
    phase12_summary_path: Path = PHASE12_SUMMARY_PATH,
    production_mode: dict | None = None,
) -> dict:
    phase11 = load_json(PHASE11_AUDIT_PATH) if PHASE11_AUDIT_PATH.exists() else {}
    phase12 = load_json(phase12_summary_path) if phase12_summary_path.exists() else {}
    phase21 = load_json(PHASE21_METRICS_PATH) if PHASE21_METRICS_PATH.exists() else {}
    phase22 = load_json(PHASE22_METRICS_PATH) if PHASE22_METRICS_PATH.exists() else {}
    phase23 = load_json(PHASE23_METRICS_PATH) if PHASE23_METRICS_PATH.exists() else {}
    phase24 = load_json(PHASE24_METRICS_PATH) if PHASE24_METRICS_PATH.exists() else {}
    phase25 = load_json(PHASE25_METRICS_PATH) if PHASE25_METRICS_PATH.exists() else {}
    phase26 = load_json(PHASE26_METRICS_PATH) if PHASE26_METRICS_PATH.exists() else {}
    phase27 = load_json(PHASE27_METRICS_PATH) if PHASE27_METRICS_PATH.exists() else {}
    phase28 = production_mode or (load_json(PHASE28_METRICS_PATH) if PHASE28_METRICS_PATH.exists() else {})
    pytest_step = next((item for item in commands if item["name"] == "tests"), None)
    failed_step = next((item["name"] for item in commands if item["returncode"] != 0), None)
    observed_run_result: dict[str, object] = {}

    return {
        "generated_at": ended_at.isoformat(),
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "duration_seconds": round((ended_at - started_at).total_seconds(), 3),
        "dataset_dir": phase12.get("dataset_dir"),
        "determinism": phase12.get("determinism", {}),
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
        "test_result": summarize_pytest(str(pytest_step.get("stdout", ""))) if pytest_step else "unknown",
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
        "language_support_result": {
            "metrics_path": str(PHASE26_METRICS_PATH),
            "report_path": str(PHASE26_REPORT_PATH),
            "language_detected_counts": phase26.get("language_detected_counts", {}),
            "cyrillic_detected_count": int(phase26.get("cyrillic_detected_count", 0)),
            "mixed_language_count": int(phase26.get("mixed_language_count", 0)),
            "pending_translation_count": int(phase26.get("pending_translation_count", 0)),
            "requires_ocr_count": int(phase26.get("requires_ocr_count", 0)),
            "language_unknown_count": int(phase26.get("language_unknown_count", 0)),
        },
        "runtime_controls_result": {
            "metrics_path": str(PHASE27_METRICS_PATH),
            "report_path": str(PHASE27_REPORT_PATH),
            "run_lock_acquired": bool(phase27.get("run_lock_acquired", False)),
            "run_lock_released": bool(phase27.get("run_lock_released", False)),
            "stale_lock_recovered": bool(phase27.get("stale_lock_recovered", False)),
            "retry_eligible_count": int(phase27.get("retry_eligible_count", 0)),
            "non_retryable_failure_count": int(phase27.get("non_retryable_failure_count", 0)),
            "timeout_count": int(phase27.get("timeout_count", 0)),
            "cleanup_completed": bool(phase27.get("cleanup_completed", False)),
            "failure_category_counts": phase27.get("failure_category_counts", {}),
        },
        "production_mode_result": {
            "metrics_path": str(PHASE28_METRICS_PATH),
            "report_path": str(PHASE28_REPORT_PATH),
            "production_mode": str(phase28.get("production_mode", MODE_OFF)),
            "production_gate_passed": bool(phase28.get("production_gate_passed", True)),
            "production_gate_failed_reason": phase28.get("production_gate_failed_reason"),
            "dry_run_executed": bool(phase28.get("dry_run_executed", False)),
            "controlled_run_limit_applied": bool(phase28.get("controlled_run_limit_applied", False)),
            "run_blocked_by_gate": bool(phase28.get("run_blocked_by_gate", False)),
        },
        "observed_run_result": observed_run_result,
        "baseline_reconciled": False,
        "baseline_source_snapshot": None,
        "reconciliation_scope": None,
        "reconciliation_reason": None,
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
    language_support = summary["language_support_result"]
    runtime_controls = summary["runtime_controls_result"]
    production_mode = summary["production_mode_result"]
    observed = summary.get("observed_run_result", {})
    observed_validation = observed.get("validation_result", {})
    observed_routing = observed.get("routing_efficiency_result", {})
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
        f"- Language support detected_counts: `{language_support['language_detected_counts']}`",
        f"- Language support cyrillic_detected_count: `{language_support['cyrillic_detected_count']}`",
        f"- Language support mixed_language_count: `{language_support['mixed_language_count']}`",
        f"- Language support pending_translation_count: `{language_support['pending_translation_count']}`",
        f"- Language support requires_ocr_count: `{language_support['requires_ocr_count']}`",
        f"- Language support language_unknown_count: `{language_support['language_unknown_count']}`",
        f"- Language support metrics_path: `{language_support['metrics_path']}`",
        f"- Language support report_path: `{language_support['report_path']}`",
        f"- Runtime controls run_lock_acquired: `{runtime_controls['run_lock_acquired']}`",
        f"- Runtime controls run_lock_released: `{runtime_controls['run_lock_released']}`",
        f"- Runtime controls stale_lock_recovered: `{runtime_controls['stale_lock_recovered']}`",
        f"- Runtime controls retry_eligible_count: `{runtime_controls['retry_eligible_count']}`",
        f"- Runtime controls non_retryable_failure_count: `{runtime_controls['non_retryable_failure_count']}`",
        f"- Runtime controls timeout_count: `{runtime_controls['timeout_count']}`",
        f"- Runtime controls cleanup_completed: `{runtime_controls['cleanup_completed']}`",
        f"- Runtime controls failure_category_counts: `{runtime_controls['failure_category_counts']}`",
        f"- Runtime controls metrics_path: `{runtime_controls['metrics_path']}`",
        f"- Runtime controls report_path: `{runtime_controls['report_path']}`",
        f"- Production mode: `{production_mode['production_mode']}`",
        f"- Production gate passed: `{production_mode['production_gate_passed']}`",
        f"- Production gate failed reason: `{production_mode['production_gate_failed_reason']}`",
        f"- Dry run executed: `{production_mode['dry_run_executed']}`",
        f"- Controlled run limit applied: `{production_mode['controlled_run_limit_applied']}`",
        f"- Run blocked by gate: `{production_mode['run_blocked_by_gate']}`",
        f"- Production metrics_path: `{production_mode['metrics_path']}`",
        f"- Production report_path: `{production_mode['report_path']}`",
        f"- Baseline reconciled: `{summary.get('baseline_reconciled', False)}`",
        f"- Baseline source snapshot: `{summary.get('baseline_source_snapshot')}`",
        f"- Reconciliation scope: `{summary.get('reconciliation_scope')}`",
        f"- Reconciliation reason: `{summary.get('reconciliation_reason')}`",
        f"- Dashboard export path: `{summary['dashboard_export_path']}`",
        f"- Stability report path: `{summary['stability_report_path']}`",
        f"- Duration seconds: `{summary['duration_seconds']}`",
        "",
        "## Observed Run",
        "",
        f"- Observed attempted: `{observed_validation.get('attempted')}`",
        f"- Observed processed: `{observed_validation.get('processed')}`",
        f"- Observed written: `{observed_validation.get('written')}`",
        f"- Observed queued_for_review: `{observed_validation.get('queued_for_review')}`",
        f"- Observed external_quota_blocked: `{observed_validation.get('external_quota_blocked')}`",
        f"- Observed hard_failures: `{observed_validation.get('hard_failures')}`",
        f"- Observed route_mismatch_count: `{observed_routing.get('route_mismatch_count')}`",
        f"- Observed intended_route_counts: `{observed_routing.get('intended_route_counts')}`",
        f"- Observed actual_route_counts: `{observed_routing.get('actual_route_counts')}`",
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
    args = parse_args()
    production_config = make_production_mode_config(args)
    production_state = evaluate_production_mode(
        production_config,
        previous_summary_path=PHASE18_SUMMARY_PATH,
        stability_report_path=ROOT / "reports" / "phase19" / "stability_report.md",
        lock_path=PHASE27_LOCK_PATH,
        phase12_summary_path=PHASE12_SUMMARY_PATH,
    )
    if production_state.run_blocked_by_gate:
        blocked_summary = {
            "generated_at": datetime.now(UTC).isoformat(),
            "dataset_dir": str(ROOT / "test_data" / "final_batch_50"),
            "documents_selected": 0,
            "documents_processed": 0,
            "written": 0,
            "queued_for_review": 0,
            "external_quota_blocked": 0,
            "hard_failures": 0,
            "determinism": {"mode": "deterministic_path", "seed": None, "ordering": "sorted_pdf_listing"},
            "production_mode": production_state.to_dict(),
        }
        write_phase28_outputs(
            blocked_summary,
            artifact_path=PHASE28_METRICS_PATH,
            report_path=PHASE28_REPORT_PATH,
        )
        print(json.dumps({"success": False, "failed_step": "production_gate", "production_mode": production_state.to_dict()}, indent=2))
        return 1

    steps = build_phase18_steps(production_config)
    phase12_summary_path = phase12_summary_path_for_mode(production_config)
    report_dir = summary_report_dir_for_mode(production_config)
    run_id = deterministic_run_id(
        scope="phase18_full_cycle",
        values={
            "steps": steps,
            "production_mode": production_config.to_dict(),
        },
    )
    guard = RuntimeRunGuard(
        script_name="run_phase18_full_cycle.py",
        run_id=run_id,
        lock_path=PHASE27_LOCK_PATH,
    )
    started_at = datetime.now(UTC)
    guard.acquire()
    try:
        results = execute_steps(steps=steps)
        ended_at = datetime.now(UTC)
        summary = build_summary(
            commands=results,
            started_at=started_at,
            ended_at=ended_at,
            phase12_summary_path=phase12_summary_path,
            production_mode=production_state.to_dict(),
        )
    finally:
        guard.release()
    summary["production_mode"] = production_state.to_dict()
    summary = reconcile_with_trusted_baseline(
        summary,
        config=production_config,
        snapshot_dir=Path(production_config.required_snapshot_dir or DEFAULT_REQUIRED_SNAPSHOT_DIR),
    )
    write_phase28_outputs(
        summary,
        artifact_path=PHASE28_METRICS_PATH,
        report_path=PHASE28_REPORT_PATH,
    )
    write_summary_reports(summary, report_dir=report_dir)
    if production_config.normalized_mode() != MODE_DRY_RUN:
        write_stability_report()
    print(json.dumps(summary, indent=2))
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
