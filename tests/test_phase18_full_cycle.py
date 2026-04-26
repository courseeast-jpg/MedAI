from __future__ import annotations

from pathlib import Path

from app import config as app_config
from scripts.run_phase18_full_cycle import (
    PHASE18_STEPS,
    build_summary,
    execute_steps,
    write_summary_reports,
)


def test_phase18_command_list_order():
    names = [name for name, _ in PHASE18_STEPS]
    commands = [command for _, command in PHASE18_STEPS]

    assert names == [
        "tests",
        "phase11_audit",
        "validation",
        "dashboard_latest",
        "dashboard_export",
    ]
    assert commands[0][-2:] == ["pytest", "tests"]
    assert commands[1][-1] == "scripts\\run_phase11_integration_audit.py"
    assert commands[2][-3:] == ["--dataset-dir", "test_data\\final_batch_50", "--quota-safe"]
    assert commands[3][-1] == "--latest"
    assert commands[4][-1] == "--export"


def test_report_writer_works(tmp_path: Path):
    summary = {
        "generated_at": "2026-04-24T22:00:00+00:00",
        "started_at": "2026-04-24T21:59:00+00:00",
        "ended_at": "2026-04-24T22:00:00+00:00",
        "duration_seconds": 60.0,
        "commit_hash": "abc123",
        "git_status": "clean",
        "steps": [
            {"name": "tests", "command": ["python", "-m", "pytest", "tests"], "returncode": 0},
            {"name": "phase11_audit", "command": ["python", "scripts\\run_phase11_integration_audit.py"], "returncode": 0},
        ],
        "success": True,
        "failed_step": None,
        "test_result": "150 passed",
        "phase11_audit_result": "passed",
        "validation_result": {
            "attempted": 50,
            "processed": 46,
            "written": 46,
            "queued_for_review": 0,
            "external_quota_blocked": 4,
            "hard_failures": 0,
            "avg_confidence": 0.7,
            "review_queue_items": 4,
            "review_queue_path": "artifacts/phase12_real_world_validation/runtime/review_queue.jsonl",
        },
        "observability_result": {
            "metrics_path": "artifacts/phase21/observability_metrics.json",
            "report_path": "reports/phase21/observability_report.md",
            "route_mismatch_count": 1,
            "low_confidence_count": 2,
            "quota_safe_block_count": 4,
            "extractor_route_counts": {"spacy": 46},
            "extractor_actual_counts": {"spacy": 46},
            "per_stage_duration_ms": {"extraction": {"avg_duration_ms": 25.0}},
        },
        "calibration_result": {
            "metrics_path": "artifacts/phase22/confidence_calibration.json",
            "report_path": "reports/phase22/accuracy_calibration_report.md",
            "average_raw_confidence": 0.7,
            "average_calibrated_confidence": 0.7,
            "confidence_band_counts": {"acceptable": 46},
            "review_recommendation_counts": {"accept": 46},
            "route_mismatch_count": 1,
        },
        "routing_efficiency_result": {
            "metrics_path": "artifacts/phase23/routing_efficiency.json",
            "report_path": "reports/phase23/routing_efficiency_report.md",
            "intended_route_counts": {"spacy": 46},
            "actual_route_counts": {"spacy": 46},
            "route_mismatch_count": 1,
            "quota_block_avoided_count": 0,
            "total_estimated_cost_units": 0.0,
            "total_saved_cost_units": 0.0,
        },
        "semantic_enrichment_result": {
            "metrics_path": "artifacts/phase24/semantic_enrichment.json",
            "report_path": "reports/phase24/semantic_enrichment_report.md",
            "enrichment_applied_count": 46,
            "negation_detected_count": 2,
            "temporal_detected_count": 4,
            "relationships_detected_count": 3,
        },
        "medical_coding_result": {
            "metrics_path": "artifacts/phase25/medical_coding.json",
            "report_path": "reports/phase25/medical_coding_report.md",
            "coding_attempted_count": 90,
            "coding_success_count": 12,
            "coding_unmapped_count": 70,
            "coding_ambiguous_count": 1,
            "coding_skipped_count": 7,
            "coding_status_counts": {"ambiguous": 1, "coded": 12, "skipped": 7, "unmapped": 70},
        },
        "language_support_result": {
            "metrics_path": "artifacts/phase26/language_support.json",
            "report_path": "reports/phase26/language_support_report.md",
            "language_detected_counts": {"english": 46},
            "cyrillic_detected_count": 0,
            "mixed_language_count": 0,
            "pending_translation_count": 0,
            "requires_ocr_count": 0,
            "language_unknown_count": 0,
        },
        "runtime_controls_result": {
            "metrics_path": "artifacts/phase27/runtime_controls.json",
            "report_path": "reports/phase27/production_hardening_report.md",
            "run_lock_acquired": True,
            "run_lock_released": True,
            "stale_lock_recovered": False,
            "retry_eligible_count": 4,
            "non_retryable_failure_count": 1,
            "timeout_count": 0,
            "cleanup_completed": False,
            "failure_category_counts": {"external_quota_block": 4, "none": 45, "operator_review_required": 1},
        },
        "production_mode_result": {
            "metrics_path": "artifacts/phase28/production_mode.json",
            "report_path": "reports/phase28/production_readiness_report.md",
            "production_mode": "OFF",
            "production_gate_passed": True,
            "production_gate_failed_reason": None,
            "dry_run_executed": False,
            "controlled_run_limit_applied": False,
            "run_blocked_by_gate": False,
        },
        "dashboard_export_path": "reports/phase17/dashboard_latest.md",
        "stability_report_path": "reports/phase19/stability_report.md",
    }

    json_path, md_path = write_summary_reports(summary, report_dir=tmp_path)

    assert json_path.exists()
    assert md_path.exists()
    assert '"success": true' in json_path.read_text(encoding="utf-8").lower()
    assert "Phase 18 Full Cycle Summary" in md_path.read_text(encoding="utf-8")


def test_failure_handling_returns_nonzero_and_stops():
    calls: list[list[str]] = []

    def fake_runner(command: list[str]) -> dict:
        calls.append(command)
        if len(calls) == 2:
            return {"command": command, "returncode": 1, "stdout": "", "stderr": "failed"}
        return {"command": command, "returncode": 0, "stdout": "", "stderr": ""}

    results = execute_steps(runner=fake_runner)

    assert len(results) == 2
    assert results[-1]["returncode"] == 1
    assert calls == [command for _, command in PHASE18_STEPS[:2]]


def test_no_pipeline_configuration_is_mutated(tmp_path: Path):
    before = {
        "ENABLE_HYPOTHESIS_TIER": app_config.ENABLE_HYPOTHESIS_TIER,
        "ENABLE_TRUTH_RESOLUTION": app_config.ENABLE_TRUTH_RESOLUTION,
        "ENABLE_DECISION_SCORING": app_config.ENABLE_DECISION_SCORING,
        "EXTRACTION_ACCEPT_THRESHOLD": app_config.EXTRACTION_ACCEPT_THRESHOLD,
        "EXTRACTION_REVIEW_THRESHOLD": app_config.EXTRACTION_REVIEW_THRESHOLD,
    }

    summary = {
        "generated_at": "2026-04-24T22:00:00+00:00",
        "started_at": "2026-04-24T21:59:00+00:00",
        "ended_at": "2026-04-24T22:00:00+00:00",
        "duration_seconds": 60.0,
        "commit_hash": "abc123",
        "git_status": "clean",
        "steps": [],
        "success": True,
        "failed_step": None,
        "test_result": "150 passed",
        "phase11_audit_result": "passed",
        "validation_result": {
            "attempted": 50,
            "processed": 46,
            "written": 46,
            "queued_for_review": 0,
            "external_quota_blocked": 4,
            "hard_failures": 0,
            "avg_confidence": 0.7,
            "review_queue_items": 4,
            "review_queue_path": "artifacts/phase12_real_world_validation/runtime/review_queue.jsonl",
        },
        "observability_result": {
            "metrics_path": "artifacts/phase21/observability_metrics.json",
            "report_path": "reports/phase21/observability_report.md",
            "route_mismatch_count": 1,
            "low_confidence_count": 2,
            "quota_safe_block_count": 4,
            "extractor_route_counts": {"spacy": 46},
            "extractor_actual_counts": {"spacy": 46},
            "per_stage_duration_ms": {"extraction": {"avg_duration_ms": 25.0}},
        },
        "calibration_result": {
            "metrics_path": "artifacts/phase22/confidence_calibration.json",
            "report_path": "reports/phase22/accuracy_calibration_report.md",
            "average_raw_confidence": 0.7,
            "average_calibrated_confidence": 0.7,
            "confidence_band_counts": {"acceptable": 46},
            "review_recommendation_counts": {"accept": 46},
            "route_mismatch_count": 1,
        },
        "routing_efficiency_result": {
            "metrics_path": "artifacts/phase23/routing_efficiency.json",
            "report_path": "reports/phase23/routing_efficiency_report.md",
            "intended_route_counts": {"spacy": 46},
            "actual_route_counts": {"spacy": 46},
            "route_mismatch_count": 1,
            "quota_block_avoided_count": 0,
            "total_estimated_cost_units": 0.0,
            "total_saved_cost_units": 0.0,
        },
        "semantic_enrichment_result": {
            "metrics_path": "artifacts/phase24/semantic_enrichment.json",
            "report_path": "reports/phase24/semantic_enrichment_report.md",
            "enrichment_applied_count": 46,
            "negation_detected_count": 2,
            "temporal_detected_count": 4,
            "relationships_detected_count": 3,
        },
        "medical_coding_result": {
            "metrics_path": "artifacts/phase25/medical_coding.json",
            "report_path": "reports/phase25/medical_coding_report.md",
            "coding_attempted_count": 90,
            "coding_success_count": 12,
            "coding_unmapped_count": 70,
            "coding_ambiguous_count": 1,
            "coding_skipped_count": 7,
            "coding_status_counts": {"ambiguous": 1, "coded": 12, "skipped": 7, "unmapped": 70},
        },
        "language_support_result": {
            "metrics_path": "artifacts/phase26/language_support.json",
            "report_path": "reports/phase26/language_support_report.md",
            "language_detected_counts": {"english": 46},
            "cyrillic_detected_count": 0,
            "mixed_language_count": 0,
            "pending_translation_count": 0,
            "requires_ocr_count": 0,
            "language_unknown_count": 0,
        },
        "runtime_controls_result": {
            "metrics_path": "artifacts/phase27/runtime_controls.json",
            "report_path": "reports/phase27/production_hardening_report.md",
            "run_lock_acquired": True,
            "run_lock_released": True,
            "stale_lock_recovered": False,
            "retry_eligible_count": 4,
            "non_retryable_failure_count": 1,
            "timeout_count": 0,
            "cleanup_completed": False,
            "failure_category_counts": {"external_quota_block": 4, "none": 45, "operator_review_required": 1},
        },
        "production_mode_result": {
            "metrics_path": "artifacts/phase28/production_mode.json",
            "report_path": "reports/phase28/production_readiness_report.md",
            "production_mode": "OFF",
            "production_gate_passed": True,
            "production_gate_failed_reason": None,
            "dry_run_executed": False,
            "controlled_run_limit_applied": False,
            "run_blocked_by_gate": False,
        },
        "dashboard_export_path": "reports/phase17/dashboard_latest.md",
        "stability_report_path": "reports/phase19/stability_report.md",
    }

    write_summary_reports(summary, report_dir=tmp_path)

    after = {
        "ENABLE_HYPOTHESIS_TIER": app_config.ENABLE_HYPOTHESIS_TIER,
        "ENABLE_TRUTH_RESOLUTION": app_config.ENABLE_TRUTH_RESOLUTION,
        "ENABLE_DECISION_SCORING": app_config.ENABLE_DECISION_SCORING,
        "EXTRACTION_ACCEPT_THRESHOLD": app_config.EXTRACTION_ACCEPT_THRESHOLD,
        "EXTRACTION_REVIEW_THRESHOLD": app_config.EXTRACTION_REVIEW_THRESHOLD,
    }

    assert after == before


def test_successful_full_cycle_summary_remains_passing_with_phase21_observability(tmp_path: Path, monkeypatch):
    from scripts import run_phase18_full_cycle as phase18

    phase11_path = tmp_path / "phase11.json"
    phase12_path = tmp_path / "phase12.json"
    phase21_path = tmp_path / "phase21.json"
    phase22_path = tmp_path / "phase22.json"
    phase23_path = tmp_path / "phase23.json"
    phase24_path = tmp_path / "phase24.json"
    phase25_path = tmp_path / "phase25.json"
    phase26_path = tmp_path / "phase26.json"
    phase27_path = tmp_path / "phase27.json"
    phase28_path = tmp_path / "phase28.json"
    phase11_path.write_text('{"merge_recommended": true}', encoding="utf-8")
    phase12_path.write_text(
        """{
  "documents_selected": 50,
  "documents_processed": 46,
  "written": 46,
  "queued_for_review": 0,
  "external_quota_blocked": 4,
  "hard_failures": 0,
  "review_queue": {"items": 30, "path": "runtime/review_queue.jsonl"},
  "aggregate": {"avg_confidence": 0.7}
}""",
        encoding="utf-8",
    )
    phase21_path.write_text(
        """{
  "route_mismatch_count": 1,
  "low_confidence_count": 2,
  "quota_safe_block_count": 4,
  "extractor_route_counts": {"spacy": 46},
  "extractor_actual_counts": {"spacy": 46},
  "per_stage_duration_ms": {"extraction": {"avg_duration_ms": 25.0}}
}""",
        encoding="utf-8",
    )
    phase22_path.write_text(
        """{
  "average_raw_confidence": 0.7,
  "average_calibrated_confidence": 0.7,
  "confidence_band_counts": {"acceptable": 46},
  "review_recommendation_counts": {"accept": 46},
  "route_mismatch_count": 1
}""",
        encoding="utf-8",
    )
    phase23_path.write_text(
        """{
  "intended_route_counts": {"spacy": 46},
  "actual_route_counts": {"spacy": 46},
  "route_mismatch_count": 1,
  "quota_block_avoided_count": 0,
  "total_estimated_cost_units": 0.0,
  "total_saved_cost_units": 0.0
}""",
        encoding="utf-8",
    )
    phase24_path.write_text(
        """{
  "enrichment_applied_count": 46,
  "negation_detected_count": 2,
  "temporal_detected_count": 4,
  "relationships_detected_count": 3
}""",
        encoding="utf-8",
    )
    phase25_path.write_text(
        """{
  "coding_attempted_count": 90,
  "coding_success_count": 12,
  "coding_unmapped_count": 70,
  "coding_ambiguous_count": 1,
  "coding_skipped_count": 7,
  "coding_status_counts": {"ambiguous": 1, "coded": 12, "skipped": 7, "unmapped": 70}
}""",
        encoding="utf-8",
    )
    phase26_path.write_text(
        """{
  "language_detected_counts": {"english": 46},
  "cyrillic_detected_count": 0,
  "mixed_language_count": 0,
  "pending_translation_count": 0,
  "requires_ocr_count": 0,
  "language_unknown_count": 0
}""",
        encoding="utf-8",
    )
    phase27_path.write_text(
        """{
  "run_lock_acquired": true,
  "run_lock_released": true,
  "stale_lock_recovered": false,
  "retry_eligible_count": 4,
  "non_retryable_failure_count": 1,
  "timeout_count": 0,
  "cleanup_completed": false,
  "failure_category_counts": {"external_quota_block": 4, "none": 45, "operator_review_required": 1}
}""",
        encoding="utf-8",
    )
    phase28_path.write_text(
        """{
  "production_mode": "OFF",
  "production_gate_passed": true,
  "production_gate_failed_reason": null,
  "dry_run_executed": false,
  "controlled_run_limit_applied": false,
  "run_blocked_by_gate": false
}""",
        encoding="utf-8",
    )
    monkeypatch.setattr(phase18, "PHASE11_AUDIT_PATH", phase11_path)
    monkeypatch.setattr(phase18, "PHASE12_SUMMARY_PATH", phase12_path)
    monkeypatch.setattr(phase18, "PHASE21_METRICS_PATH", phase21_path)
    monkeypatch.setattr(phase18, "PHASE22_METRICS_PATH", phase22_path)
    monkeypatch.setattr(phase18, "PHASE23_METRICS_PATH", phase23_path)
    monkeypatch.setattr(phase18, "PHASE24_METRICS_PATH", phase24_path)
    monkeypatch.setattr(phase18, "PHASE25_METRICS_PATH", phase25_path)
    monkeypatch.setattr(phase18, "PHASE26_METRICS_PATH", phase26_path)
    monkeypatch.setattr(phase18, "PHASE27_METRICS_PATH", phase27_path)
    monkeypatch.setattr(phase18, "PHASE28_METRICS_PATH", phase28_path)

    summary = build_summary(
        commands=[{"name": "tests", "command": ["python", "-m", "pytest", "tests"], "returncode": 0, "stdout": "=== 159 passed ===", "stderr": ""}],
        started_at=phase18.datetime.fromisoformat("2026-04-24T21:59:00+00:00"),
        ended_at=phase18.datetime.fromisoformat("2026-04-24T22:00:00+00:00"),
    )

    assert summary["success"] is True
    assert summary["validation_result"]["hard_failures"] == 0
    assert summary["observability_result"]["quota_safe_block_count"] == 4
    assert summary["calibration_result"]["confidence_band_counts"] == {"acceptable": 46}
    assert summary["routing_efficiency_result"]["actual_route_counts"] == {"spacy": 46}
    assert summary["semantic_enrichment_result"]["enrichment_applied_count"] == 46
    assert summary["medical_coding_result"]["coding_success_count"] == 12
    assert summary["language_support_result"]["language_detected_counts"] == {"english": 46}
    assert summary["runtime_controls_result"]["run_lock_acquired"] is True
    assert summary["production_mode_result"]["production_mode"] == "OFF"


def test_full_cycle_summary_is_deterministic_with_phase25_metrics(tmp_path: Path, monkeypatch):
    from scripts import run_phase18_full_cycle as phase18

    phase11_path = tmp_path / "phase11.json"
    phase12_path = tmp_path / "phase12.json"
    phase21_path = tmp_path / "phase21.json"
    phase22_path = tmp_path / "phase22.json"
    phase23_path = tmp_path / "phase23.json"
    phase24_path = tmp_path / "phase24.json"
    phase25_path = tmp_path / "phase25.json"
    phase26_path = tmp_path / "phase26.json"
    phase27_path = tmp_path / "phase27.json"
    phase28_path = tmp_path / "phase28.json"
    phase11_path.write_text('{"merge_recommended": true}', encoding="utf-8")
    phase12_path.write_text(
        '{"documents_selected": 50, "documents_processed": 46, "written": 45, "queued_for_review": 1, "external_quota_blocked": 4, "hard_failures": 0, "review_queue": {"items": 31, "path": "runtime/review_queue.jsonl"}, "aggregate": {"avg_confidence": 0.692}}',
        encoding="utf-8",
    )
    phase21_path.write_text(
        '{"route_mismatch_count": 1, "low_confidence_count": 1, "quota_safe_block_count": 4, "extractor_route_counts": {"phi3": 1, "spacy": 45}, "extractor_actual_counts": {"phi3": 1, "spacy": 45}, "per_stage_duration_ms": {"medical_coding": {"avg_duration_ms": 0.0}}}',
        encoding="utf-8",
    )
    phase22_path.write_text(
        '{"average_raw_confidence": 0.692, "average_calibrated_confidence": 0.692, "confidence_band_counts": {"acceptable": 45, "reject": 1}, "review_recommendation_counts": {"accept": 44, "accept_with_route_audit": 1, "reject_do_not_write": 1}, "route_mismatch_count": 1}',
        encoding="utf-8",
    )
    phase23_path.write_text(
        '{"intended_route_counts": {"gemini": 1, "phi3": 1, "spacy": 44}, "actual_route_counts": {"phi3": 1, "spacy": 45}, "route_mismatch_count": 1, "quota_block_avoided_count": 1, "total_estimated_cost_units": 0.005, "total_saved_cost_units": 0.02}',
        encoding="utf-8",
    )
    phase24_path.write_text(
        '{"enrichment_applied_count": 45, "negation_detected_count": 0, "temporal_detected_count": 0, "relationships_detected_count": 0}',
        encoding="utf-8",
    )
    phase25_path.write_text(
        '{"coding_attempted_count": 86, "coding_success_count": 12, "coding_unmapped_count": 70, "coding_ambiguous_count": 1, "coding_skipped_count": 3, "coding_status_counts": {"ambiguous": 1, "coded": 12, "skipped": 3, "unmapped": 70}}',
        encoding="utf-8",
    )
    phase26_path.write_text(
        '{"language_detected_counts": {"english": 45}, "cyrillic_detected_count": 0, "mixed_language_count": 0, "pending_translation_count": 0, "requires_ocr_count": 0, "language_unknown_count": 0}',
        encoding="utf-8",
    )
    phase27_path.write_text(
        '{"run_lock_acquired": true, "run_lock_released": true, "stale_lock_recovered": false, "retry_eligible_count": 4, "non_retryable_failure_count": 1, "timeout_count": 0, "cleanup_completed": false, "failure_category_counts": {"external_quota_block": 4, "none": 45, "operator_review_required": 1}}',
        encoding="utf-8",
    )
    phase28_path.write_text(
        '{"production_mode": "OFF", "production_gate_passed": true, "production_gate_failed_reason": null, "dry_run_executed": false, "controlled_run_limit_applied": false, "run_blocked_by_gate": false}',
        encoding="utf-8",
    )
    monkeypatch.setattr(phase18, "PHASE11_AUDIT_PATH", phase11_path)
    monkeypatch.setattr(phase18, "PHASE12_SUMMARY_PATH", phase12_path)
    monkeypatch.setattr(phase18, "PHASE21_METRICS_PATH", phase21_path)
    monkeypatch.setattr(phase18, "PHASE22_METRICS_PATH", phase22_path)
    monkeypatch.setattr(phase18, "PHASE23_METRICS_PATH", phase23_path)
    monkeypatch.setattr(phase18, "PHASE24_METRICS_PATH", phase24_path)
    monkeypatch.setattr(phase18, "PHASE25_METRICS_PATH", phase25_path)
    monkeypatch.setattr(phase18, "PHASE26_METRICS_PATH", phase26_path)
    monkeypatch.setattr(phase18, "PHASE27_METRICS_PATH", phase27_path)
    monkeypatch.setattr(phase18, "PHASE28_METRICS_PATH", phase28_path)

    commands = [{"name": "tests", "command": ["python", "-m", "pytest", "tests"], "returncode": 0, "stdout": "=== 190 passed ===", "stderr": ""}]
    started_at = phase18.datetime.fromisoformat("2026-04-25T00:00:00+00:00")
    ended_at = phase18.datetime.fromisoformat("2026-04-25T00:05:00+00:00")

    first = build_summary(commands=commands, started_at=started_at, ended_at=ended_at)
    second = build_summary(commands=commands, started_at=started_at, ended_at=ended_at)

    assert first == second
    assert first["language_support_result"]["language_detected_counts"] == {"english": 45}
    assert first["runtime_controls_result"]["retry_eligible_count"] == 4
    assert first["production_mode_result"]["production_gate_passed"] is True
