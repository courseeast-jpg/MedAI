from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from monitoring.metrics_collector import (
    append_run_history,
    build_run_record,
    load_run_history,
)


def make_summary() -> dict:
    return {
        "generated_at": "2026-04-24T21:24:02.521094+00:00",
        "dataset_dir": "test_data\\final_batch_50",
        "documents_selected": 50,
        "documents_processed": 46,
        "written": 46,
        "queued_for_review": 0,
        "external_quota_blocked": 4,
        "hard_failures": 0,
        "review_queue": {"items": 30},
        "aggregate": {
            "outcomes": {"written": 33, "written_with_review": 13},
            "avg_confidence": 0.7,
            "review_reasons": {"clear": 13, "quarantined": 13},
        },
        "documents": [
            {"processing_time_ms": 1000.0},
            {"processing_time_ms": 2500.0},
        ],
    }


def make_aggregate() -> dict:
    return {
        "dataset_dir": "test_data\\final_batch_50",
        "documents_attempted": 50,
        "documents_processed": 46,
        "documents_quota_blocked": 4,
        "written": 46,
        "queued_for_review": 0,
        "hard_failures": 0,
        "avg_confidence_processed_only": 0.7,
        "route_distribution_actual": {"spacy": 46},
        "route_distribution_requested": {"gemini": 1, "spacy": 45, "unknown": 4},
    }


def test_run_record_computed_from_existing_artifacts_shape():
    phase21 = {
        "review_queue_items": 30,
        "route_mismatch_count": 1,
        "low_confidence_count": 2,
        "quota_safe_block_count": 4,
    }
    record = build_run_record(make_summary(), make_aggregate(), phase21_metrics=phase21)

    assert record.dataset == "test_data\\final_batch_50"
    assert record.attempted == 50
    assert record.processed == 46
    assert record.written == 46
    assert record.written_with_review == 13
    assert record.review_queue_items == 30
    assert record.external_quota_blocked == 4
    assert record.hard_failures == 0
    assert record.avg_confidence == 0.7
    assert record.route_distribution_requested == {"gemini": 1, "spacy": 45, "unknown": 4}
    assert record.route_distribution_actual == {"spacy": 46}
    assert record.route_mismatch_count == 1
    assert record.low_confidence_count == 2
    assert record.quota_safe_block_count == 4
    assert record.review_counts == {"clear": 13, "quarantined": 13}
    assert record.duration_sec == 3.5


def test_run_history_append_works(tmp_path: Path):
    history_path = tmp_path / "run_history.jsonl"
    first = build_run_record(make_summary(), make_aggregate())
    second_summary = make_summary()
    second_summary["generated_at"] = "2026-04-24T22:24:02.521094+00:00"
    second_summary["written"] = 47
    second_aggregate = make_aggregate()
    second_aggregate["written"] = 47
    second = build_run_record(second_summary, second_aggregate)

    append_run_history(first, history_path=history_path)
    append_run_history(second, history_path=history_path)
    history = load_run_history(history_path=history_path)

    assert len(history) == 2
    assert history[0]["run_id"] == first.run_id
    assert history[1]["run_id"] == second.run_id
    assert history[1]["written"] == 47


def test_dashboard_renders_without_error(tmp_path: Path):
    history_path = tmp_path / "run_history.jsonl"
    first = build_run_record(make_summary(), make_aggregate())
    second_summary = make_summary()
    second_summary["generated_at"] = "2026-04-24T22:24:02.521094+00:00"
    second_summary["written"] = 45
    second_aggregate = make_aggregate()
    second_aggregate["written"] = 45
    second = build_run_record(second_summary, second_aggregate)
    append_run_history(first, history_path=history_path)
    append_run_history(second, history_path=history_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts\\run_phase17_dashboard.py",
            "--latest",
            "--history-path",
            str(history_path),
        ],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Phase 17 Latest Run" in result.stdout
    assert "d_written" in result.stdout

    export_result = subprocess.run(
        [
            sys.executable,
            "scripts\\run_phase17_dashboard.py",
            "--export",
            "--history-path",
            str(history_path),
        ],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        check=False,
    )

    assert export_result.returncode == 0
    default_export = Path(__file__).resolve().parent.parent / "reports" / "phase17" / "dashboard_latest.md"
    assert default_export.exists()
    exported_text = default_export.read_text(encoding="utf-8")
    assert "Phase 17 Dashboard" in exported_text
