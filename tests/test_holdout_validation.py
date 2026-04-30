from __future__ import annotations

import json
from pathlib import Path

import scripts.run_holdout_validation as holdout


def configure_paths(monkeypatch, tmp_path: Path) -> None:
    input_dir = tmp_path / "holdout_validation_input"
    report_dir = tmp_path / "reports" / "holdout_validation"
    baseline_dir = tmp_path / "reports" / "baseline_phase33"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_file = baseline_dir / "baseline_metrics.json"
    baseline_file.write_text(
        json.dumps({
            "commit_hash": "503479a",
            "total_files": 15,
            "accepted": 11,
            "review": 4,
            "empty": 1,
            "test_pass": True,
            "timestamp": "2026-04-30T18:56:39.6547109-04:00",
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(holdout, "HOLDOUT_INPUT_DIR", input_dir)
    monkeypatch.setattr(holdout, "HOLDOUT_REPORT_DIR", report_dir)
    monkeypatch.setattr(holdout, "HOLDOUT_ARCHIVE_DIR", report_dir / "archive")
    monkeypatch.setattr(holdout, "HOLDOUT_REVIEW_DIR", report_dir / "review")
    monkeypatch.setattr(holdout, "HOLDOUT_ERROR_DIR", report_dir / "error")
    monkeypatch.setattr(holdout, "HOLDOUT_JSON_REPORT", report_dir / "latest_holdout_validation.json")
    monkeypatch.setattr(holdout, "HOLDOUT_MD_REPORT", report_dir / "latest_holdout_validation.md")
    monkeypatch.setattr(holdout, "HOLDOUT_REVIEW_AUDIT_JSON", report_dir / "review_audit.json")
    monkeypatch.setattr(holdout, "HOLDOUT_REVIEW_AUDIT_MD", report_dir / "review_audit.md")
    monkeypatch.setattr(holdout, "HOLDOUT_COMPARISON_REPORT", report_dir / "comparison_to_baseline.md")
    monkeypatch.setattr(holdout, "BASELINE_JSON_REPORT", baseline_file)


def test_holdout_empty_input_exits_cleanly_and_writes_reports(monkeypatch, tmp_path: Path) -> None:
    configure_paths(monkeypatch, tmp_path)

    summary = holdout.run_holdout_validation()

    assert summary["total_files"] == 0
    assert summary["accepted_count"] == 0
    assert summary["review_count"] == 0
    assert summary["error_count"] == 0
    assert holdout.HOLDOUT_JSON_REPORT.exists()
    assert holdout.HOLDOUT_MD_REPORT.exists()
    assert holdout.HOLDOUT_COMPARISON_REPORT.exists()
    report = holdout.HOLDOUT_COMPARISON_REPORT.read_text(encoding="utf-8")
    assert "Baseline commit: `503479a`" in report
