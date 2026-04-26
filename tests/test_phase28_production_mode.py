from __future__ import annotations

import json
from pathlib import Path

from execution.production_mode import (
    MODE_CONTROLLED,
    MODE_DRY_RUN,
    MODE_LIVE,
    MODE_OFF,
    ProductionModeConfig,
    evaluate_production_mode,
)
from monitoring.observability import build_phase28_metrics, write_phase28_outputs
from scripts.run_phase18_full_cycle import (
    PHASE18_STEPS,
    build_phase18_steps,
    phase12_summary_path_for_mode,
    reconcile_with_trusted_baseline,
    summary_report_dir_for_mode,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _gate_inputs(tmp_path: Path, *, review_queue_items: int = 1) -> dict[str, Path]:
    previous_summary = tmp_path / "reports" / "phase18" / "full_cycle_summary.json"
    stability_report = tmp_path / "reports" / "phase19" / "stability_report.md"
    phase12_summary = tmp_path / "artifacts" / "phase12_real_world_validation" / "phase12_real_world_validation_summary.json"
    snapshot_dir = tmp_path / "phase27_continuation_20260424"
    snapshot_zip = tmp_path / "phase27_continuation_20260424.zip"
    snapshot_dir.mkdir(parents=True)
    snapshot_zip.write_text("zip", encoding="utf-8")
    _write_json(previous_summary, {"success": True, "failed_step": None})
    stability_report.parent.mkdir(parents=True, exist_ok=True)
    stability_report.write_text("# Phase 19 Stability Report\n\n- Overall status: `STABLE`\n", encoding="utf-8")
    _write_json(phase12_summary, {"review_queue": {"items": review_queue_items}})
    return {
        "previous_summary": previous_summary,
        "stability_report": stability_report,
        "phase12_summary": phase12_summary,
        "snapshot_dir": snapshot_dir,
        "snapshot_zip": snapshot_zip,
        "lock_path": tmp_path / "artifacts" / "phase27" / "full_cycle_run.lock",
    }


def test_dry_run_does_not_target_canonical_outputs():
    config = ProductionModeConfig(mode=MODE_DRY_RUN, require_snapshot_before_run=True)

    steps = build_phase18_steps(config)

    assert steps != PHASE18_STEPS
    validation_command = dict(steps)["validation"]
    assert "--output-dir" in validation_command
    assert "phase28" in " ".join(validation_command).lower()
    assert "phase28" in str(phase12_summary_path_for_mode(config)).lower()
    assert "phase28" in str(summary_report_dir_for_mode(config)).lower()


def test_controlled_mode_enforces_limit():
    config = ProductionModeConfig(mode=MODE_CONTROLLED, max_documents_per_run=5)

    validation_command = dict(build_phase18_steps(config))["validation"]

    assert validation_command[-2:] == ["--limit", "5"]


def test_live_mode_requires_all_gates_to_pass(tmp_path: Path):
    inputs = _gate_inputs(tmp_path)
    config = ProductionModeConfig(
        mode=MODE_LIVE,
        audit_required=True,
        require_snapshot_before_run=True,
        run_approval=False,
        review_queue_acknowledged=True,
        required_snapshot_dir=str(inputs["snapshot_dir"]),
        required_snapshot_zip=str(inputs["snapshot_zip"]),
    )

    state = evaluate_production_mode(
        config,
        previous_summary_path=inputs["previous_summary"],
        stability_report_path=inputs["stability_report"],
        lock_path=inputs["lock_path"],
        phase12_summary_path=inputs["phase12_summary"],
    )

    assert state.production_gate_passed is False
    assert state.production_gate_failed_reason == "run_approval_missing"


def test_run_is_blocked_if_snapshot_missing(tmp_path: Path):
    inputs = _gate_inputs(tmp_path)
    config = ProductionModeConfig(
        mode=MODE_CONTROLLED,
        require_snapshot_before_run=True,
        review_queue_acknowledged=True,
        required_snapshot_dir=str(tmp_path / "missing_snapshot"),
        required_snapshot_zip=str(tmp_path / "missing_snapshot.zip"),
    )

    state = evaluate_production_mode(
        config,
        previous_summary_path=inputs["previous_summary"],
        stability_report_path=inputs["stability_report"],
        lock_path=inputs["lock_path"],
        phase12_summary_path=inputs["phase12_summary"],
    )

    assert state.production_gate_passed is False
    assert state.production_gate_failed_reason == "snapshot_missing"


def test_run_is_blocked_if_lock_exists(tmp_path: Path):
    inputs = _gate_inputs(tmp_path)
    inputs["lock_path"].parent.mkdir(parents=True, exist_ok=True)
    inputs["lock_path"].write_text("locked", encoding="utf-8")
    config = ProductionModeConfig(
        mode=MODE_CONTROLLED,
        review_queue_acknowledged=True,
        required_snapshot_dir=str(inputs["snapshot_dir"]),
        required_snapshot_zip=str(inputs["snapshot_zip"]),
    )

    state = evaluate_production_mode(
        config,
        previous_summary_path=inputs["previous_summary"],
        stability_report_path=inputs["stability_report"],
        lock_path=inputs["lock_path"],
        phase12_summary_path=inputs["phase12_summary"],
    )

    assert state.production_gate_passed is False
    assert state.production_gate_failed_reason == "runtime_lock_present"


def test_run_is_blocked_if_nondeterminism_detected(tmp_path: Path):
    inputs = _gate_inputs(tmp_path)
    inputs["stability_report"].write_text("# Phase 19 Stability Report\n\n- Overall status: `UNSTABLE`\n", encoding="utf-8")
    config = ProductionModeConfig(
        mode=MODE_CONTROLLED,
        review_queue_acknowledged=True,
        required_snapshot_dir=str(inputs["snapshot_dir"]),
        required_snapshot_zip=str(inputs["snapshot_zip"]),
    )

    state = evaluate_production_mode(
        config,
        previous_summary_path=inputs["previous_summary"],
        stability_report_path=inputs["stability_report"],
        lock_path=inputs["lock_path"],
        phase12_summary_path=inputs["phase12_summary"],
    )

    assert state.production_gate_passed is False
    assert state.production_gate_failed_reason == "determinism_not_verified"


def test_review_queue_must_be_acknowledged_in_controlled_and_live_modes(tmp_path: Path):
    inputs = _gate_inputs(tmp_path, review_queue_items=31)
    config = ProductionModeConfig(
        mode=MODE_CONTROLLED,
        required_snapshot_dir=str(inputs["snapshot_dir"]),
        required_snapshot_zip=str(inputs["snapshot_zip"]),
    )

    state = evaluate_production_mode(
        config,
        previous_summary_path=inputs["previous_summary"],
        stability_report_path=inputs["stability_report"],
        lock_path=inputs["lock_path"],
        phase12_summary_path=inputs["phase12_summary"],
    )

    assert state.production_gate_passed is False
    assert state.production_gate_failed_reason == "review_queue_unacknowledged"


def test_phase27_baseline_behavior_unchanged_in_off_mode(tmp_path: Path):
    inputs = _gate_inputs(tmp_path, review_queue_items=31)
    config = ProductionModeConfig(mode=MODE_OFF)

    state = evaluate_production_mode(
        config,
        previous_summary_path=inputs["previous_summary"],
        stability_report_path=inputs["stability_report"],
        lock_path=inputs["lock_path"],
        phase12_summary_path=inputs["phase12_summary"],
    )

    assert build_phase18_steps(config) == PHASE18_STEPS
    assert state.production_gate_passed is True
    assert state.production_mode == MODE_OFF


def test_phase28_outputs_are_written(tmp_path: Path):
    artifact_path = tmp_path / "artifacts" / "phase28" / "production_mode.json"
    report_path = tmp_path / "reports" / "phase28" / "production_readiness_report.md"
    summary = {
        "generated_at": "2026-04-25T18:00:00+00:00",
        "dataset_dir": "test_data\\final_batch_50",
        "documents_selected": 50,
        "documents_processed": 46,
        "written": 45,
        "queued_for_review": 1,
        "external_quota_blocked": 4,
        "hard_failures": 0,
        "determinism": {"mode": "deterministic_path", "seed": None, "ordering": "sorted_pdf_listing"},
        "production_mode": {
            "production_mode": "OFF",
            "production_gate_passed": True,
            "production_gate_failed_reason": None,
            "dry_run_executed": False,
            "controlled_run_limit_applied": False,
            "run_blocked_by_gate": False,
            "max_documents_per_run": 0,
            "max_concurrent_runs": 1,
            "audit_required": False,
            "require_snapshot_before_run": False,
            "run_approval": False,
            "review_queue_acknowledged": False,
            "required_snapshot_dir": None,
            "required_snapshot_zip": None,
            "previous_run_completed_cleanly": True,
            "deterministic_outputs_verified": True,
            "unresolved_runtime_lock": False,
            "snapshot_verified": False,
            "audit_report_available": True,
            "review_queue_items": 31,
        },
        "observed_run_result": {
            "validation_result": {
                "attempted": 50,
                "processed": 47,
                "written": 46,
                "queued_for_review": 1,
                "external_quota_blocked": 3,
                "hard_failures": 0,
                "avg_confidence": 0.7,
                "review_queue_items": 31,
            },
            "observability_result": {
                "route_mismatch_count": 2,
                "low_confidence_count": 1,
                "quota_safe_block_count": 3,
                "extractor_route_counts": {"gemini": 1, "spacy": 45},
                "extractor_actual_counts": {"phi3": 1, "spacy": 45},
            },
            "routing_efficiency_result": {
                "route_mismatch_count": 2,
                "intended_route_counts": {"gemini": 1, "spacy": 45},
                "actual_route_counts": {"phi3": 1, "spacy": 45},
                "quota_block_avoided_count": 1,
                "total_estimated_cost_units": 0.005,
                "total_saved_cost_units": 0.02,
            },
        },
        "baseline_reconciled": True,
        "baseline_source_snapshot": "C:\\baseline",
        "reconciliation_scope": "reporting_and_artifact_reconciliation_only",
        "reconciliation_reason": "observed_validation_drift",
    }

    metrics = write_phase28_outputs(summary, artifact_path=artifact_path, report_path=report_path)

    assert artifact_path.exists()
    assert report_path.exists()
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == metrics
    assert build_phase28_metrics(summary)["production_mode"] == "OFF"
    assert metrics["observed_validation_result"]["processed"] == 47
    assert metrics["canonical_validation_result"]["processed"] == 46
    report_text = report_path.read_text(encoding="utf-8")
    assert "Observed validation result" in report_text
    assert "Canonical validation result" in report_text
    assert "drift remain visible" in report_text


def test_reconciliation_preserves_observed_drift_visibility(tmp_path: Path, monkeypatch):
    snapshot_dir = tmp_path / "snapshot"
    snapshot_phase12 = snapshot_dir / "artifacts" / "phase12_real_world_validation" / "phase12_real_world_validation_summary.json"
    snapshot_phase12.parent.mkdir(parents=True, exist_ok=True)
    snapshot_phase12.write_text(
        json.dumps(
            {
                "documents_selected": 50,
                "documents_processed": 46,
                "written": 45,
                "queued_for_review": 1,
                "external_quota_blocked": 4,
                "hard_failures": 0,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "scripts.run_phase18_full_cycle.restore_trusted_baseline_outputs",
        lambda _: None,
    )
    monkeypatch.setattr(
        "scripts.run_phase18_full_cycle.build_summary",
        lambda **_: {
            "generated_at": "2026-04-26T03:13:52+00:00",
            "started_at": "2026-04-26T03:00:00+00:00",
            "ended_at": "2026-04-26T03:13:52+00:00",
            "duration_seconds": 832.0,
            "validation_result": {
                "attempted": 50,
                "processed": 46,
                "written": 45,
                "queued_for_review": 1,
                "external_quota_blocked": 4,
                "hard_failures": 0,
            },
            "observability_result": {
                "route_mismatch_count": 1,
                "low_confidence_count": 1,
                "quota_safe_block_count": 4,
                "extractor_route_counts": {"spacy": 45, "phi3": 1},
                "extractor_actual_counts": {"spacy": 45, "phi3": 1},
            },
            "calibration_result": {},
            "routing_efficiency_result": {
                "route_mismatch_count": 1,
                "intended_route_counts": {"gemini": 1, "spacy": 45},
                "actual_route_counts": {"phi3": 1, "spacy": 45},
                "quota_block_avoided_count": 0,
                "total_estimated_cost_units": 0.0,
                "total_saved_cost_units": 0.0,
            },
            "production_mode_result": {"production_mode": "OFF"},
        },
    )
    summary = {
        "started_at": "2026-04-26T03:00:00+00:00",
        "ended_at": "2026-04-26T03:13:52+00:00",
        "steps": [],
        "production_mode": {"production_mode": "OFF"},
        "validation_result": {
            "attempted": 50,
            "processed": 47,
            "written": 46,
            "queued_for_review": 1,
            "external_quota_blocked": 3,
            "hard_failures": 0,
        },
        "observability_result": {
            "route_mismatch_count": 2,
            "low_confidence_count": 1,
            "quota_safe_block_count": 3,
            "extractor_route_counts": {"gemini": 1, "spacy": 45},
            "extractor_actual_counts": {"phi3": 1, "spacy": 45},
        },
        "calibration_result": {},
        "routing_efficiency_result": {
            "route_mismatch_count": 2,
            "intended_route_counts": {"gemini": 2, "spacy": 44},
            "actual_route_counts": {"phi3": 2, "spacy": 44},
            "quota_block_avoided_count": 1,
            "total_estimated_cost_units": 0.005,
            "total_saved_cost_units": 0.02,
        },
    }

    reconciled = reconcile_with_trusted_baseline(
        summary,
        config=ProductionModeConfig(mode=MODE_OFF),
        snapshot_dir=snapshot_dir,
    )

    assert reconciled["baseline_reconciled"] is True
    assert reconciled["reconciliation_reason"] == "observed_validation_drift"
    assert reconciled["observed_run_result"]["validation_result"]["processed"] == 47
    assert reconciled["validation_result"]["processed"] == 46
    assert reconciled["observed_run_result"]["routing_efficiency_result"]["route_mismatch_count"] == 2
    assert reconciled["routing_efficiency_result"]["route_mismatch_count"] == 1
