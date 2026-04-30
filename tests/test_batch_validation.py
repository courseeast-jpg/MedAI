from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import scripts.run_batch_validation as batch


@dataclass
class FakeResult:
    outcome: str = "written"
    validation_status: str = "accepted"
    validation_errors: list[dict] = field(default_factory=list)
    extractor_result: dict = field(default_factory=dict)
    audit: dict = field(default_factory=dict)


class FakePipeline:
    def process_text(self, text: str, *, specialty: str, source_name: str, session_id: str) -> FakeResult:
        del text, specialty, source_name, session_id
        return FakeResult(
            extractor_result={
                "entities": [{"type": "diagnosis", "text": "diabetes"}],
                "actual_extractor": "spacy",
                "primary_extractor": "gemini",
                "fallback_extractor": "spacy",
                "fallback_reason": "gemini_quota_or_rate_limit",
                "terminal_empty_prevented": True,
                "confidence": 0.72,
            }
        )


def configure_paths(monkeypatch, tmp_path: Path) -> None:
    input_dir = tmp_path / "real_validation_input"
    report_dir = tmp_path / "reports" / "batch_validation"
    monkeypatch.setattr(batch, "REAL_VALIDATION_INPUT_DIR", input_dir)
    monkeypatch.setattr(batch, "BATCH_REPORT_DIR", report_dir)
    monkeypatch.setattr(batch, "BATCH_ARCHIVE_DIR", report_dir / "archive")
    monkeypatch.setattr(batch, "BATCH_REVIEW_DIR", report_dir / "review")
    monkeypatch.setattr(batch, "BATCH_ERROR_DIR", report_dir / "error")
    monkeypatch.setattr(batch, "BATCH_JSON_REPORT", report_dir / "latest_batch_validation.json")
    monkeypatch.setattr(batch, "BATCH_MD_REPORT", report_dir / "latest_batch_validation.md")


def test_empty_input_folder_exits_cleanly_and_writes_reports(monkeypatch, tmp_path: Path) -> None:
    configure_paths(monkeypatch, tmp_path)

    summary = batch.run_batch_validation(pipeline=FakePipeline())

    assert summary["total_files"] == 0
    assert summary["accepted_count"] == 0
    assert summary["error_count"] == 0
    assert batch.BATCH_JSON_REPORT.exists()
    assert batch.BATCH_MD_REPORT.exists()


def test_batch_validation_report_schema_contains_required_fields(monkeypatch, tmp_path: Path) -> None:
    configure_paths(monkeypatch, tmp_path)
    batch.ensure_batch_validation_dirs()
    source = batch.REAL_VALIDATION_INPUT_DIR / "sample.txt"
    source.write_text("Patient has diabetes.", encoding="utf-8")

    summary = batch.run_batch_validation(pipeline=FakePipeline())

    assert summary["total_files"] == 1
    assert summary["accepted_count"] == 1
    assert summary["fallback_count"] == 1
    result = summary["results"][0]
    for field in batch.REQUIRED_RESULT_FIELDS:
        assert field in result
    assert result["entity_count"] == 1
    assert result["fallback_reason"] == "gemini_quota_or_rate_limit"
    assert result["terminal_empty_prevented"] is True
    assert (batch.BATCH_ARCHIVE_DIR / "sample.txt").exists()
    assert source.exists()

