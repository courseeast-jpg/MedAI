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
    monkeypatch.setattr(phase18, "PHASE11_AUDIT_PATH", phase11_path)
    monkeypatch.setattr(phase18, "PHASE12_SUMMARY_PATH", phase12_path)
    monkeypatch.setattr(phase18, "PHASE21_METRICS_PATH", phase21_path)

    summary = build_summary(
        commands=[{"name": "tests", "command": ["python", "-m", "pytest", "tests"], "returncode": 0, "stdout": "=== 159 passed ===", "stderr": ""}],
        started_at=phase18.datetime.fromisoformat("2026-04-24T21:59:00+00:00"),
        ended_at=phase18.datetime.fromisoformat("2026-04-24T22:00:00+00:00"),
    )

    assert summary["success"] is True
    assert summary["validation_result"]["hard_failures"] == 0
    assert summary["observability_result"]["quota_safe_block_count"] == 4
