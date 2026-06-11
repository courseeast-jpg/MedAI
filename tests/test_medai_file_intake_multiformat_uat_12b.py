from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import app.test_launcher as launcher
from app.main import queue_display_state, visible_current_run


SUPPORTED_NAMES = [
    "sample.pdf",
    "sample.txt",
    "sample.png",
    "sample.jpg",
    "sample.jpeg",
    "sample.tif",
    "sample.tiff",
    "sample.bmp",
    "sample.docx",
]


@dataclass
class FakeResult:
    outcome: str = "queued_for_review"
    validation_status: str = "needs_review"
    validation_errors: list[dict] = field(default_factory=list)
    extractor_result: dict = field(default_factory=lambda: {"actual_extractor": "local_text", "confidence": 0.71})
    audit: dict = field(default_factory=dict)


class FakePipeline:
    def __init__(self) -> None:
        self.pdf_calls: list[Path] = []
        self.text_calls: list[dict] = []

    def process_pdf(self, source_path: Path, *, specialty: str, session_id: str) -> FakeResult:
        self.pdf_calls.append(source_path)
        return FakeResult(extractor_result={"actual_extractor": "pdf_pipeline", "confidence": 0.7})

    def process_text(self, text: str, *, specialty: str, source_name: str, session_id: str) -> FakeResult:
        self.text_calls.append({"text": text, "source_name": source_name})
        return FakeResult(extractor_result={"actual_extractor": "text_pipeline", "confidence": 0.72})


class AcceptingTextPipeline(FakePipeline):
    def process_text(self, text: str, *, specialty: str, source_name: str, session_id: str) -> FakeResult:
        self.text_calls.append({"text": text, "source_name": source_name})
        return FakeResult(
            outcome="written",
            validation_status="accepted",
            extractor_result={"actual_extractor": "text_pipeline", "confidence": 0.91},
        )


def _write_docx(path: Path, text: str = "local synthetic text") -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "word/document.xml",
            (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                f"<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>"
            ),
        )


def _configure_launcher_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launcher, "TEST_INPUT_DIR", tmp_path / "test_input")
    monkeypatch.setattr(launcher, "TEST_REVIEW_DIR", tmp_path / "test_review")
    monkeypatch.setattr(launcher, "TEST_ARCHIVE_DIR", tmp_path / "test_archive")
    monkeypatch.setattr(launcher, "TEST_OUTPUT_DIR", tmp_path / "test_output")
    monkeypatch.setattr(launcher, "TEST_RUN_REPORT_DIR", tmp_path / "reports" / "test_runs")
    monkeypatch.setattr(launcher, "LATEST_JSON_REPORT", tmp_path / "reports" / "test_runs" / "latest_test_run.json")
    monkeypatch.setattr(launcher, "LATEST_MD_REPORT", tmp_path / "reports" / "test_runs" / "latest_test_run.md")


def test_supported_multiformat_names_are_accepted() -> None:
    for name in SUPPORTED_NAMES:
        assert launcher.safe_test_filename(name) == name


def test_unsupported_extension_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported test file type"):
        launcher.safe_test_filename("sample.exe")


def test_list_input_files_includes_only_supported_formats(tmp_path: Path) -> None:
    input_dir = tmp_path / "test_input"
    input_dir.mkdir()
    for name in SUPPORTED_NAMES:
        (input_dir / name).write_bytes(b"synthetic")
    (input_dir / "ignore.csv").write_text("leave", encoding="utf-8")

    listed = launcher.list_test_input_files(input_dir)

    assert [path.name for path in listed] == sorted(SUPPORTED_NAMES)


def test_queue_state_enables_start_only_after_local_queue_exists() -> None:
    selected = queue_display_state(queued_count=0, selected_count=2)
    queued = queue_display_state(queued_count=2, selected_count=0)
    empty = queue_display_state(queued_count=0, selected_count=0)

    assert selected["message"] == "Files selected; adding to local queue..."
    assert selected["start_enabled"] is False
    assert queued["message"] == "Ready to process 2 file(s)."
    assert queued["start_enabled"] is True
    assert empty["message"] == "No documents queued."
    assert empty["start_enabled"] is False


def test_stale_failed_run_is_hidden_when_queue_is_empty() -> None:
    active_run = {"failed": True, "error_count": 1}

    assert visible_current_run(active_run, queued_count=0, selected_count=0) is None
    assert visible_current_run(active_run, queued_count=1, selected_count=0) == active_run


def test_pdf_txt_docx_dispatch_to_existing_local_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _configure_launcher_dirs(monkeypatch, tmp_path)
    pipeline = FakePipeline()
    pdf_path = tmp_path / "sample.pdf"
    txt_path = tmp_path / "sample.txt"
    docx_path = tmp_path / "sample.docx"
    pdf_path.write_bytes(b"%PDF-1.4")
    txt_path.write_text("local text", encoding="utf-8")
    _write_docx(docx_path, "docx text")

    pdf_result = launcher._process_one_file(pipeline, pdf_path, specialty="general", run_id="run")
    txt_result = launcher._process_one_file(pipeline, txt_path, specialty="general", run_id="run")
    docx_result = launcher._process_one_file(pipeline, docx_path, specialty="general", run_id="run")

    assert pdf_result.status == "review"
    assert txt_result.status == "review"
    assert docx_result.status == "review"
    assert [path.name for path in pipeline.pdf_calls] == ["sample.pdf"]
    assert [call["source_name"] for call in pipeline.text_calls] == ["sample.txt", "sample.docx"]
    assert pipeline.text_calls[1]["text"] == "docx text"


def test_image_dispatch_is_review_bound_when_local_ocr_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _configure_launcher_dirs(monkeypatch, tmp_path)
    monkeypatch.setattr(
        launcher,
        "recover_text_from_image_local",
        lambda path: launcher.LocalImageOcrResult(
            available=False,
            attempted=False,
            engine="tesseract_local",
            text_visibility="unavailable",
            error_bucket="local_ocr_unavailable",
        ),
    )
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    result = launcher._process_one_file(FakePipeline(), image_path, specialty="general", run_id="run")

    assert result.status == "review"
    assert result.outcome == "queued_for_review"
    assert result.selected_extractor == "local_image_ocr_unavailable"
    assert result.ocr_gate_auto_accept_allowed is False
    assert result.error == "Local image OCR extractor is safely unavailable in this runtime."
    assert (tmp_path / "test_review" / "sample.png").exists()


def test_docx_dispatch_is_forced_review_bound_even_if_text_pipeline_accepts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _configure_launcher_dirs(monkeypatch, tmp_path)
    docx_path = tmp_path / "sample.docx"
    _write_docx(docx_path, "docx text")

    result = launcher._process_one_file(AcceptingTextPipeline(), docx_path, specialty="general", run_id="run")

    assert result.status == "review"
    assert result.outcome == "queued_for_review"
    assert result.validation_status == "needs_review"
    assert (tmp_path / "test_review" / "sample.docx").exists()
