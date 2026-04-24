from __future__ import annotations

import json
from pathlib import Path

from app import config as app_config
from monitoring.run_comparator import compare_last_runs, compare_runs, load_tolerances, write_stability_report


def write_history(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def make_run(run_id: str, *, processed: int, written: int, quota: int, avg_conf: float) -> dict:
    return {
        "run_id": run_id,
        "timestamp": "2026-04-24T21:00:00+00:00",
        "dataset": "test_data\\final_batch_50",
        "attempted": 50,
        "processed": processed,
        "written": written,
        "written_with_review": 13,
        "external_quota_blocked": quota,
        "hard_failures": 0,
        "avg_confidence": avg_conf,
        "route_distribution_requested": {"spacy": 45, "gemini": 1, "unknown": 4},
        "route_distribution_actual": {"spacy": processed},
        "review_counts": {"clear": 13, "quarantined": 13},
        "duration_sec": 10.0,
        "determinism": {"mode": "deterministic_path", "seed": None},
        "quota_behavior": {"external_attempts": quota, "skipped_external_quota": quota, "reasons": {"external_quota": quota}},
    }


def test_comparator_works(tmp_path: Path):
    tolerances_path = tmp_path / "phase19_stability.yaml"
    tolerances_path.write_text(
        "\n".join([
            "processed_delta_max: 2",
            "written_delta_max: 2",
            "queued_delta_max: 2",
            "confidence_delta_max: 0.05",
        ]) + "\n",
        encoding="utf-8",
    )
    comparison = compare_runs(
        make_run("run-2", processed=46, written=46, quota=4, avg_conf=0.7),
        make_run("run-1", processed=46, written=45, quota=4, avg_conf=0.68),
        load_tolerances(tolerances_path),
    )

    assert comparison["deltas"]["written"] == 1
    assert comparison["deltas"]["queued_for_review"] == -1
    assert comparison["status"] == "STABLE"


def test_tolerance_flags_trigger_correctly(tmp_path: Path):
    tolerances_path = tmp_path / "phase19_stability.yaml"
    tolerances_path.write_text(
        "\n".join([
            "processed_delta_max: 1",
            "written_delta_max: 1",
            "queued_delta_max: 1",
            "confidence_delta_max: 0.01",
        ]) + "\n",
        encoding="utf-8",
    )
    history_path = tmp_path / "run_history.jsonl"
    write_history(
        history_path,
        [
            make_run("run-1", processed=46, written=46, quota=4, avg_conf=0.7),
            make_run("run-2", processed=43, written=42, quota=7, avg_conf=0.62),
        ],
    )

    bundle = compare_last_runs(history_path=history_path, config_path=tolerances_path, limit=2)

    assert bundle["status"] == "UNSTABLE"
    assert bundle["comparisons"][-1]["exceeded"]["processed"] is True
    assert bundle["comparisons"][-1]["exceeded"]["written"] is True
    assert bundle["comparisons"][-1]["exceeded"]["avg_confidence"] is True


def test_stability_report_writer_and_no_pipeline_mutation(tmp_path: Path):
    before = {
        "ENABLE_HYPOTHESIS_TIER": app_config.ENABLE_HYPOTHESIS_TIER,
        "ENABLE_TRUTH_RESOLUTION": app_config.ENABLE_TRUTH_RESOLUTION,
        "ENABLE_DECISION_SCORING": app_config.ENABLE_DECISION_SCORING,
        "EXTRACTION_ACCEPT_THRESHOLD": app_config.EXTRACTION_ACCEPT_THRESHOLD,
        "EXTRACTION_REVIEW_THRESHOLD": app_config.EXTRACTION_REVIEW_THRESHOLD,
    }
    tolerances_path = tmp_path / "phase19_stability.yaml"
    tolerances_path.write_text(
        "\n".join([
            "processed_delta_max: 2",
            "written_delta_max: 2",
            "queued_delta_max: 2",
            "confidence_delta_max: 0.05",
        ]) + "\n",
        encoding="utf-8",
    )
    history_path = tmp_path / "run_history.jsonl"
    write_history(
        history_path,
        [
            make_run("run-1", processed=46, written=46, quota=4, avg_conf=0.7),
            make_run("run-2", processed=47, written=46, quota=3, avg_conf=0.7),
        ],
    )

    report_path = tmp_path / "stability_report.md"
    written_path = write_stability_report(history_path=history_path, config_path=tolerances_path, report_path=report_path, limit=2)

    assert written_path.exists()
    assert "Phase 19 Stability Report" in written_path.read_text(encoding="utf-8")

    after = {
        "ENABLE_HYPOTHESIS_TIER": app_config.ENABLE_HYPOTHESIS_TIER,
        "ENABLE_TRUTH_RESOLUTION": app_config.ENABLE_TRUTH_RESOLUTION,
        "ENABLE_DECISION_SCORING": app_config.ENABLE_DECISION_SCORING,
        "EXTRACTION_ACCEPT_THRESHOLD": app_config.EXTRACTION_ACCEPT_THRESHOLD,
        "EXTRACTION_REVIEW_THRESHOLD": app_config.EXTRACTION_REVIEW_THRESHOLD,
    }
    assert after == before
