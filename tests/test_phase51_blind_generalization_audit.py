from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from scripts.run_phase51_blind_pdf_generalization_audit import (
    OPERATOR_SUMMARY,
    PRIVATE_MAPPING,
    ensure_real_validation_input,
    run_audit,
)


class FakeBlindPipeline:
    def __init__(self):
        self.router = SimpleNamespace(gemini_quota_blocked=False)

    def run(self, job):
        return SimpleNamespace(
            outcome="queued_for_review",
            validation_status="needs_review",
            validation_errors=[{"code": "confidence_below_threshold"}],
            extractor_result={
                "extractor": "spacy",
                "actual_extractor": "spacy",
                "entities": [],
                "confidence": 0.42,
                "latency_ms": 1,
                "raw_text": "",
                "notes": [],
            },
            audit={"extractor_actual": "spacy", "confidence": 0.42},
        )


def test_phase51_real_validation_input_exists_or_is_created(tmp_path: Path):
    input_dir = tmp_path / "real_validation_input"

    ensure_real_validation_input(input_dir)

    assert input_dir.exists()
    assert (input_dir / ".gitkeep").exists()


def test_phase51_real_validation_pdfs_are_ignored_by_git():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "check-ignore", "real_validation_input/example.pdf"],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0


def test_phase51_audit_script_handles_empty_folder(tmp_path: Path):
    input_dir = tmp_path / "real_validation_input"
    report_dir = tmp_path / "reports" / "phase51"

    report = run_audit(input_dir=input_dir, report_dir=report_dir, pipeline=FakeBlindPipeline())

    assert report["total_files"] == 0
    assert report["conclusion"] == "no_input_files"
    assert (report_dir / "phase51_blind_generalization_audit_report.json").exists()
    assert (report_dir / OPERATOR_SUMMARY.name).exists()


def test_phase51_public_report_avoids_raw_text_and_raw_filenames(tmp_path: Path):
    input_dir = tmp_path / "real_validation_input"
    report_dir = tmp_path / "reports" / "phase51"
    input_dir.mkdir()
    raw_filename = "John Smith DOB 1980 labs.txt"
    raw_text = "Patient John Smith DOB 01/02/1980 MRN AB123456"
    (input_dir / raw_filename).write_text(raw_text, encoding="utf-8")

    report = run_audit(input_dir=input_dir, report_dir=report_dir, pipeline=FakeBlindPipeline())
    public_report = (report_dir / "phase51_blind_generalization_audit_report.json").read_text(encoding="utf-8")
    public_markdown = (report_dir / "phase51_blind_generalization_audit_report.md").read_text(encoding="utf-8")

    assert report["local_only_mode"] is True
    assert report["external_api_default_allowed"] is False
    assert report["external_api_used"] is False
    assert raw_filename not in public_report
    assert raw_text not in public_report
    assert "John Smith" not in public_report
    assert raw_filename not in public_markdown
    assert raw_text not in public_markdown
    assert report["results"][0]["original_filename_redacted"] == "[REDACTED]"
    assert report["results"][0]["filename_hash"]


def test_phase51_private_filename_mapping_is_ignored_and_contains_mapping(tmp_path: Path):
    input_dir = tmp_path / "real_validation_input"
    report_dir = tmp_path / "reports" / "phase51"
    input_dir.mkdir()
    (input_dir / "private-name.txt").write_text("Synthetic text", encoding="utf-8")

    run_audit(input_dir=input_dir, report_dir=report_dir, pipeline=FakeBlindPipeline())
    mapping = json.loads((report_dir / PRIVATE_MAPPING.name).read_text(encoding="utf-8"))

    assert list(mapping["files"].values())[0]["original_filename"] == "private-name.txt"

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "check-ignore", "reports/phase51_blind_generalization_audit/local_filename_mapping_PRIVATE.json"],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0


def test_phase51_no_pdfs_are_tracked_under_reports():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "ls-files", "reports"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    tracked_report_pdfs = [line for line in result.stdout.splitlines() if line.lower().endswith(".pdf")]

    assert tracked_report_pdfs == []
