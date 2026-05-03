from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from scripts.run_phase53_blind_pdf_generalization_audit import (
    JSON_REPORT,
    OPERATOR_SUMMARY,
    PRIVATE_MAPPING,
    ensure_real_validation_input,
    run_audit,
)


APP_MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"


class FakePhase53Pipeline:
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
                "selected_extractor": "spacy",
                "entities": [],
                "confidence": 0.42,
                "latency_ms": 1,
                "raw_text": "",
                "notes": [],
            },
            audit={"extractor_actual": "spacy", "confidence": 0.42},
        )


def test_phase53_empty_real_validation_input_produces_safe_no_input_report(tmp_path: Path):
    input_dir = tmp_path / "real_validation_input"
    report_dir = tmp_path / "reports" / "phase53"

    report = run_audit(input_dir=input_dir, report_dir=report_dir, pipeline=FakePhase53Pipeline())

    assert report["total_files"] == 0
    assert report["conclusion"] == "no_input_files"
    assert report["external_api_used"] is False
    assert report["raw_phi_logged_in_public_reports"] is False
    assert report["local_only_forced"] is True
    assert (report_dir / JSON_REPORT.name).exists()
    assert (report_dir / OPERATOR_SUMMARY.name).exists()


def test_phase53_public_reports_use_safe_ids_hashes_and_no_raw_filenames(tmp_path: Path):
    input_dir = tmp_path / "real_validation_input"
    report_dir = tmp_path / "reports" / "phase53"
    input_dir.mkdir()
    raw_filename = "Jane Doe DOB 1975 MRN 12345 labs.txt"
    raw_text = "Patient Jane Doe DOB 01/02/1975 MRN 12345"
    (input_dir / raw_filename).write_text(raw_text, encoding="utf-8")

    report = run_audit(input_dir=input_dir, report_dir=report_dir, pipeline=FakePhase53Pipeline())
    public_json = (report_dir / JSON_REPORT.name).read_text(encoding="utf-8")
    public_md = (report_dir / "phase53_blind_generalization_audit_report.md").read_text(encoding="utf-8")

    first = report["results"][0]
    assert first["file_id"] == "file_001"
    assert first["filename_hash"]
    assert first["content_hash"]
    assert raw_filename not in public_json
    assert raw_filename not in public_md
    assert raw_text not in public_json
    assert "Jane Doe" not in public_json
    assert "MRN 12345" not in public_md


def test_phase53_private_mapping_created_and_git_ignored(tmp_path: Path):
    input_dir = tmp_path / "real_validation_input"
    report_dir = tmp_path / "reports" / "phase53"
    input_dir.mkdir()
    (input_dir / "private-name.txt").write_text("Synthetic fixture", encoding="utf-8")

    run_audit(input_dir=input_dir, report_dir=report_dir, pipeline=FakePhase53Pipeline())
    mapping = json.loads((report_dir / PRIVATE_MAPPING.name).read_text(encoding="utf-8"))

    assert mapping["files"]["file_001"]["original_filename"] == "private-name.txt"
    assert "absolute_path" not in mapping["files"]["file_001"]

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "check-ignore", "reports/phase53_blind_generalization_audit/local_filename_mapping_PRIVATE.json"],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0


def test_phase53_real_validation_input_exists_and_pdfs_are_ignored():
    repo_root = Path(__file__).resolve().parents[1]
    ensure_real_validation_input(repo_root / "real_validation_input")

    result = subprocess.run(
        ["git", "check-ignore", "real_validation_input/example.pdf"],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0


def test_phase53_external_apis_blocked_and_local_only_forced(tmp_path: Path):
    input_dir = tmp_path / "real_validation_input"
    report_dir = tmp_path / "reports" / "phase53"
    input_dir.mkdir()
    (input_dir / "synthetic.txt").write_text("Synthetic text only", encoding="utf-8")

    report = run_audit(input_dir=input_dir, report_dir=report_dir, pipeline=FakePhase53Pipeline())

    assert report["local_only_mode"] is True
    assert report["local_only_forced"] is True
    assert report["external_api_default_allowed"] is False
    assert report["external_api_used"] is False
    assert report["results"][0]["privacy_gate_mode"] == "local_only"


def test_phase53_no_pdfs_are_tracked_under_reports():
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


def test_phase53_ui_label_references_phase53_not_prior_phase_blind_audit():
    source = APP_MAIN.read_text(encoding="utf-8")

    assert "Run Phase53 Blind Audit from real_validation_input/" in source
    assert "run_phase53_blind_pdf_generalization_audit" in source
    assert "run_phase51_blind_pdf_generalization_audit" not in source
