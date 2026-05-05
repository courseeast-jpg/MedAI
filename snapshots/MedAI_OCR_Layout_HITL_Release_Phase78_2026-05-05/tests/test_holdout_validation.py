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
    monkeypatch.setattr(holdout, "HOLDOUT_PHASE35_AUDIT_JSON", report_dir / "review_audit_phase35.json")
    monkeypatch.setattr(holdout, "HOLDOUT_PHASE35_AUDIT_MD", report_dir / "review_audit_phase35.md")
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
    assert holdout.HOLDOUT_PHASE35_AUDIT_JSON.exists()
    assert holdout.HOLDOUT_PHASE35_AUDIT_MD.exists()
    report = holdout.HOLDOUT_COMPARISON_REPORT.read_text(encoding="utf-8")
    assert "Baseline commit: `503479a`" in report


def test_phase35_review_audit_filters_review_files_and_classifies_remaining_issue(monkeypatch, tmp_path: Path) -> None:
    configure_paths(monkeypatch, tmp_path)
    holdout.ensure_holdout_dirs()
    summary = {
        "timestamp": "2026-04-30T20:00:00+00:00",
        "results": [
            {
                "filename": "accepted.pdf",
                "status": "accepted",
                "entity_count": 4,
                "entities": [{"text": "Blood"}],
                "confidence": 0.72,
                "confidence_breakdown": {"coverage": 0.4},
                "why_reviewed": [],
                "normalization_applied": False,
                "normalized_text_preview": "",
                "text_diagnostics": {
                    "preview": "Urine report with sufficient extraction",
                    "length": 200,
                    "method": "plain_text",
                    "suspicious": False,
                },
            },
            {
                "filename": "noisy.pdf",
                "status": "review",
                "entity_count": 2,
                "entities": [{"text": "Nitrite"}, {"text": "Ketones"}],
                "confidence": 0.45,
                "confidence_breakdown": {"coverage": 0.2, "entity_count": 0.3, "diversity": 0.9, "extractor_weight": 0.6},
                "why_reviewed": ["confidence_below_threshold", "low_coverage"],
                "normalization_applied": True,
                "normalized_text_preview": "Urine report Negative Yellow",
                "text_diagnostics": {
                    "preview": "UR0KULTURE |||| NEGAT1V ____ VERDHE",
                    "length": 100,
                    "method": "tesseract fallback",
                    "suspicious": True,
                },
            },
            {
                "filename": "low-coverage.pdf",
                "status": "review",
                "entity_count": 3,
                "entities": [{"text": "Klebsiella"}],
                "confidence": 0.63,
                "confidence_breakdown": {"coverage": 0.2, "entity_count": 0.6, "diversity": 1.0, "extractor_weight": 0.8},
                "why_reviewed": ["confidence_below_threshold", "low_coverage"],
                "normalization_applied": False,
                "normalized_text_preview": "",
                "text_diagnostics": {
                    "preview": "Readable specimen culture report",
                    "length": 250,
                    "method": "unknown",
                    "suspicious": False,
                },
            },
        ],
    }
    holdout.HOLDOUT_JSON_REPORT.write_text(json.dumps(summary), encoding="utf-8")

    holdout.write_phase35_review_audit()

    audit = json.loads(holdout.HOLDOUT_PHASE35_AUDIT_JSON.read_text(encoding="utf-8"))
    assert audit["total_reviewed"] == 2
    assert audit["remaining_issue_breakdown"]["ocr_noise_remaining"] == 1
    assert audit["remaining_issue_breakdown"]["low_coverage_after_normalization"] == 1
    assert {item["filename"] for item in audit["files"]} == {"noisy.pdf", "low-coverage.pdf"}
    assert audit["files"][0]["normalized_text_preview"][:300] == audit["files"][0]["normalized_text_preview"]
    assert "Remaining Issue Breakdown" in holdout.HOLDOUT_PHASE35_AUDIT_MD.read_text(encoding="utf-8")
