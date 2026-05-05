from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app.test_launcher import (
    build_test_launcher_display_state,
    clear_latest_test_reports,
    clear_test_input,
    save_uploaded_test_file,
)


class FakeUpload:
    def __init__(self, name: str, payload: bytes):
        self.name = name
        self.size = len(payload)
        self._payload = payload

    def getbuffer(self) -> memoryview:
        return memoryview(self._payload)


def test_uploaded_file_is_saved_to_test_input(tmp_path: Path):
    input_dir = tmp_path / "test_input"
    saved = save_uploaded_test_file(FakeUpload("Result Details.pdf", b"%PDF-1.4\n"), input_dir=input_dir)

    assert saved.parent == input_dir
    assert saved.name == "Result Details.pdf"
    assert saved.read_bytes() == b"%PDF-1.4\n"


def test_uploaded_duplicate_gets_safe_suffix(tmp_path: Path):
    input_dir = tmp_path / "test_input"

    first = save_uploaded_test_file(FakeUpload("same.pdf", b"one"), input_dir=input_dir)
    second = save_uploaded_test_file(FakeUpload("same.pdf", b"two"), input_dir=input_dir)

    assert first.name == "same.pdf"
    assert second.name == "same_1.pdf"
    assert first.read_bytes() == b"one"
    assert second.read_bytes() == b"two"


def test_unsupported_uploaded_file_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="Unsupported test file type"):
        save_uploaded_test_file(FakeUpload("malware.exe", b"nope"), input_dir=tmp_path / "test_input")


def test_clear_test_input_keeps_gitkeep_and_supported_scope(tmp_path: Path):
    input_dir = tmp_path / "test_input"
    input_dir.mkdir()
    (input_dir / ".gitkeep").write_text("", encoding="utf-8")
    (input_dir / "queued.pdf").write_bytes(b"%PDF")
    (input_dir / "queued.txt").write_text("hello", encoding="utf-8")
    (input_dir / "notes.tmp").write_text("leave me", encoding="utf-8")

    removed = clear_test_input(input_dir)

    assert sorted(path.name for path in removed) == ["queued.pdf", "queued.txt"]
    assert (input_dir / ".gitkeep").exists()
    assert (input_dir / "notes.tmp").exists()
    assert not (input_dir / "queued.pdf").exists()
    assert not (input_dir / "queued.txt").exists()


def test_stale_latest_report_is_not_current_when_no_files_queued(tmp_path: Path):
    latest = {"timestamp": "2026-05-01T00:00:00Z", "accepted_count": 9, "review_count": 8, "error_count": 7}

    state = build_test_launcher_display_state([], latest)

    assert state["run_status"] == "No current run"
    assert state["counter_source"] == "no current run"
    assert state["accepted"] == 0
    assert state["review"] == 0
    assert state["errors"] == 0
    assert state["show_no_supported_files"] is True


def test_latest_report_clear_does_not_delete_unrelated_reports(tmp_path: Path):
    report_dir = tmp_path / "reports" / "test_runs"
    report_dir.mkdir(parents=True)
    (report_dir / "latest_test_run.md").write_text("latest", encoding="utf-8")
    (report_dir / "latest_test_run.json").write_text("{}", encoding="utf-8")
    (report_dir / "auto_test_run.md").write_text("keep", encoding="utf-8")
    (report_dir / "latest_test_run_old.md").write_text("keep", encoding="utf-8")

    removed = clear_latest_test_reports(report_dir)

    assert sorted(path.name for path in removed) == ["latest_test_run.json", "latest_test_run.md"]
    assert not (report_dir / "latest_test_run.md").exists()
    assert not (report_dir / "latest_test_run.json").exists()
    assert (report_dir / "auto_test_run.md").exists()
    assert (report_dir / "latest_test_run_old.md").exists()


def test_no_pdfs_are_tracked_under_reports():
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
