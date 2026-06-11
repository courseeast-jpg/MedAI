from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import app.test_launcher as launcher


@dataclass
class FakeResult:
    outcome: str = "queued_for_review"
    validation_status: str = "needs_review"
    validation_errors: list[dict] = field(default_factory=list)
    extractor_result: dict = field(default_factory=lambda: {"actual_extractor": "text_pipeline", "confidence": 0.72})
    audit: dict = field(default_factory=dict)


class FakePipeline:
    def __init__(self, *, accept_text: bool = False) -> None:
        self.accept_text = accept_text
        self.pdf_calls: list[Path] = []
        self.text_calls: list[dict] = []

    def process_pdf(self, source_path: Path, *, specialty: str, session_id: str) -> FakeResult:
        self.pdf_calls.append(source_path)
        return FakeResult(extractor_result={"actual_extractor": "pdf_pipeline", "confidence": 0.7})

    def process_text(self, text: str, *, specialty: str, source_name: str, session_id: str) -> FakeResult:
        self.text_calls.append({"text": text, "source_name": source_name})
        if self.accept_text:
            return FakeResult(
                outcome="written",
                validation_status="accepted",
                extractor_result={
                    "actual_extractor": "text_pipeline",
                    "confidence": 0.95,
                    "external_api_used": False,
                    "extracted_medical_fact_count": 1,
                    "extraction_to_mkb_review_count": 1,
                },
            )
        return FakeResult()


def configure_launcher_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launcher, "TEST_INPUT_DIR", tmp_path / "test_input")
    monkeypatch.setattr(launcher, "TEST_REVIEW_DIR", tmp_path / "test_review")
    monkeypatch.setattr(launcher, "TEST_ARCHIVE_DIR", tmp_path / "test_archive")
    monkeypatch.setattr(launcher, "TEST_OUTPUT_DIR", tmp_path / "test_output")
    monkeypatch.setattr(launcher, "TEST_RUN_REPORT_DIR", tmp_path / "reports" / "test_runs")
    monkeypatch.setattr(launcher, "LATEST_JSON_REPORT", tmp_path / "reports" / "test_runs" / "latest_test_run.json")
    monkeypatch.setattr(launcher, "LATEST_MD_REPORT", tmp_path / "reports" / "test_runs" / "latest_test_run.md")


def write_docx(path: Path, text: str = "docx text") -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "word/document.xml",
            (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                f"<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>"
            ),
        )


def test_image_extension_dispatches_to_local_ocr_function(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure_launcher_dirs(monkeypatch, tmp_path)
    called: list[Path] = []

    def fake_ocr(path: Path) -> launcher.LocalImageOcrResult:
        called.append(path)
        return launcher.LocalImageOcrResult(
            available=True,
            attempted=True,
            text="Hemoglobin 13.2 g/dL",
            engine="tesseract_local",
            language="eng",
            text_visibility="recovered",
        )

    monkeypatch.setattr(launcher, "recover_text_from_image_local", fake_ocr)
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"image")

    result = launcher._process_one_file(FakePipeline(), image_path, specialty="general", run_id="run")

    assert called == [image_path]
    assert result.image_ocr_attempted is True
    assert result.image_ocr_text_visibility == "recovered"


def test_local_ocr_unavailable_returns_safe_review_bound_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configure_launcher_dirs(monkeypatch, tmp_path)
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
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"image")

    result = launcher._process_one_file(FakePipeline(), image_path, specialty="general", run_id="run")

    assert result.status == "review"
    assert result.outcome == "queued_for_review"
    assert result.selected_extractor == "local_image_ocr_unavailable"
    assert result.image_ocr_available is False
    assert result.external_api_used is False
    assert result.image_ocr_auto_accept_allowed is False


def test_local_ocr_recovered_text_routes_to_existing_text_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configure_launcher_dirs(monkeypatch, tmp_path)
    monkeypatch.setattr(
        launcher,
        "recover_text_from_image_local",
        lambda path: launcher.LocalImageOcrResult(
            available=True,
            attempted=True,
            text="Urinalysis nitrite negative",
            engine="tesseract_local",
            language="eng",
            text_visibility="recovered",
        ),
    )
    pipeline = FakePipeline()
    image_path = tmp_path / "sample.tiff"
    image_path.write_bytes(b"image")

    result = launcher._process_one_file(pipeline, image_path, specialty="general", run_id="run")

    assert pipeline.text_calls == [{"text": "Urinalysis nitrite negative", "source_name": "sample.tiff"}]
    assert result.selected_extractor == "local_image_ocr:text_pipeline"
    assert result.image_ocr_engine == "tesseract_local"


def test_image_ocr_result_is_never_auto_accepted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure_launcher_dirs(monkeypatch, tmp_path)
    monkeypatch.setattr(
        launcher,
        "recover_text_from_image_local",
        lambda path: launcher.LocalImageOcrResult(
            available=True,
            attempted=True,
            text="Glucose 5.2 mmol/L",
            engine="tesseract_local",
            language="eng",
            text_visibility="recovered",
        ),
    )
    image_path = tmp_path / "sample.bmp"
    image_path.write_bytes(b"image")

    result = launcher._process_one_file(FakePipeline(accept_text=True), image_path, specialty="general", run_id="run")

    assert result.status == "review"
    assert result.outcome == "queued_for_review"
    assert result.validation_status == "needs_review"
    assert result.image_ocr_auto_accept_allowed is False
    assert result.external_api_used is False


def test_empty_no_text_image_is_review_bound_not_accepted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure_launcher_dirs(monkeypatch, tmp_path)
    monkeypatch.setattr(
        launcher,
        "recover_text_from_image_local",
        lambda path: launcher.LocalImageOcrResult(
            available=True,
            attempted=True,
            text="",
            engine="tesseract_local",
            language="eng",
            text_visibility="not_recovered",
        ),
    )
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"image")

    result = launcher._process_one_file(FakePipeline(accept_text=True), image_path, specialty="general", run_id="run")

    assert result.status == "review_ocr_quality"
    assert result.validation_status == "empty"
    assert result.image_ocr_text_visibility == "not_recovered"
    assert result.image_ocr_auto_accept_allowed is False


def test_public_report_does_not_include_raw_ocr_text(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure_launcher_dirs(monkeypatch, tmp_path)
    monkeypatch.setattr(
        launcher,
        "recover_text_from_image_local",
        lambda path: launcher.LocalImageOcrResult(
            available=True,
            attempted=True,
            text="SECRET OCR TEXT 12345",
            engine="tesseract_local",
            language="eng",
            text_visibility="recovered",
        ),
    )
    input_dir = tmp_path / "test_input"
    input_dir.mkdir()
    (input_dir / "sample.png").write_bytes(b"image")

    launcher.run_medai_test_batch(FakePipeline(), specialty="general")

    assert "SECRET OCR TEXT 12345" not in launcher.LATEST_JSON_REPORT.read_text(encoding="utf-8")
    assert "SECRET OCR TEXT 12345" not in launcher.LATEST_MD_REPORT.read_text(encoding="utf-8")


def test_pdf_txt_docx_behavior_from_12b_remains_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configure_launcher_dirs(monkeypatch, tmp_path)
    pipeline = FakePipeline()
    pdf_path = tmp_path / "sample.pdf"
    txt_path = tmp_path / "sample.txt"
    docx_path = tmp_path / "sample.docx"
    pdf_path.write_bytes(b"%PDF-1.4")
    txt_path.write_text("txt text", encoding="utf-8")
    write_docx(docx_path)

    pdf_result = launcher._process_one_file(pipeline, pdf_path, specialty="general", run_id="run")
    txt_result = launcher._process_one_file(pipeline, txt_path, specialty="general", run_id="run")
    docx_result = launcher._process_one_file(FakePipeline(accept_text=True), docx_path, specialty="general", run_id="run")

    assert pdf_result.selected_extractor == "pdf_pipeline"
    assert txt_result.selected_extractor == "text_pipeline"
    assert docx_result.status == "review"
    assert docx_result.outcome == "queued_for_review"


def test_unsupported_extension_remains_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported test file type"):
        launcher.safe_test_filename("sample.csv")


def test_queued_image_completes_non_error_when_ocr_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configure_launcher_dirs(monkeypatch, tmp_path)
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
    input_dir = tmp_path / "test_input"
    input_dir.mkdir()
    (input_dir / "sample.png").write_bytes(b"image")

    summary = launcher.run_medai_test_batch(FakePipeline(), specialty="general")

    assert summary.error_count == 0
    assert summary.review_count == 1
