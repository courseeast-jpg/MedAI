from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from ocr_layout.image_ocr import SUPPORTED_IMAGE_EXTENSIONS, ImageOcrResult, extract_image_text
from scripts import run_phase53_blind_pdf_generalization_audit as phase53


class FakeImagePipeline:
    def __init__(self):
        self.router = SimpleNamespace(gemini_quota_blocked=False)

    def run(self, job):
        text = str(job.text or "")
        entities = [{"type": "lab", "value": "glucose"}] if "Glucose" in text else []
        confidence = 0.82 if entities else 0.42
        return SimpleNamespace(
            outcome="written" if entities else "queued_for_review",
            validation_status="accepted" if entities else "needs_review",
            validation_errors=[] if entities else [{"code": "confidence_below_threshold"}],
            extractor_result={
                "extractor": "spacy",
                "actual_extractor": "spacy",
                "selected_extractor": "spacy",
                "entities": entities,
                "confidence": confidence,
                "raw_text": "",
            },
            audit={"extractor_actual": "spacy", "confidence": confidence},
        )


def make_png(path: Path) -> None:
    image = Image.new("RGB", (120, 40), "white")
    image.save(path)


def test_phase56_supported_image_extensions_are_registered():
    expected = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    assert expected <= SUPPORTED_IMAGE_EXTENSIONS
    assert expected <= phase53.SUPPORTED_EXTENSIONS
    assert {".pdf", ".txt"} <= phase53.SUPPORTED_EXTENSIONS


def test_phase56_phase53_discovery_counts_synthetic_images(tmp_path: Path):
    input_dir = tmp_path / "real_validation_input"
    input_dir.mkdir()
    (input_dir / "one.pdf").write_bytes(b"%PDF synthetic")
    (input_dir / "two.txt").write_text("Synthetic text", encoding="utf-8")
    make_png(input_dir / "three.png")
    make_png(input_dir / "four.tif")
    (input_dir / "ignored.gif").write_bytes(b"GIF89a")

    files = phase53.supported_input_files(input_dir)

    assert len(files) == 4
    assert {path.suffix.lower() for path in files} == {".pdf", ".txt", ".png", ".tif"}


def test_phase56_synthetic_image_processes_without_external_api(tmp_path: Path, monkeypatch):
    input_dir = tmp_path / "real_validation_input"
    report_dir = tmp_path / "reports" / "phase53"
    input_dir.mkdir()
    make_png(input_dir / "visual-lab.png")

    monkeypatch.setattr(
        phase53,
        "extract_image_text",
        lambda path: ImageOcrResult(
            text="Glucose 103 mg/dL 65-99 H",
            frame_count=1,
            ocr_engine="tesseract",
            language_hint="eng",
            warnings=[],
        ),
    )
    report = phase53.run_audit(input_dir=input_dir, report_dir=report_dir, pipeline=FakeImagePipeline())

    result = report["results"][0]
    assert report["total_files"] == 1
    assert report["external_api_used"] is False
    assert result["file_type"] == "image"
    assert result["image_extension"] == ".png"
    assert result["ocr_engine"] == "tesseract"
    assert result["selected_ocr_engine"] == "image_ocr_tesseract"


def test_phase56_corrupt_image_does_not_crash_and_routes_safely(tmp_path: Path):
    input_dir = tmp_path / "real_validation_input"
    report_dir = tmp_path / "reports" / "phase53"
    input_dir.mkdir()
    (input_dir / "broken.tif").write_bytes(b"not an image")

    report = phase53.run_audit(input_dir=input_dir, report_dir=report_dir, pipeline=FakeImagePipeline())

    result = report["results"][0]
    assert result["status"] in {"review", "review_ocr_quality", "error"}
    assert result["status"] != "accepted"
    assert report["external_api_used"] is False


def test_phase56_public_reports_exclude_raw_image_filename_and_ocr_text(tmp_path: Path, monkeypatch):
    input_dir = tmp_path / "real_validation_input"
    report_dir = tmp_path / "reports" / "phase53"
    input_dir.mkdir()
    raw_filename = "Patient Jane Doe DOB 1970 scan.png"
    extracted_text = "Patient Jane Doe DOB 01/02/1970 MRN ABC123 Glucose 103"
    make_png(input_dir / raw_filename)

    monkeypatch.setattr(
        phase53,
        "extract_image_text",
        lambda path: ImageOcrResult(
            text=extracted_text,
            frame_count=1,
            ocr_engine="tesseract",
            language_hint="eng",
            warnings=[],
        ),
    )
    report = phase53.run_audit(input_dir=input_dir, report_dir=report_dir, pipeline=FakeImagePipeline())
    public_json = (report_dir / phase53.JSON_REPORT.name).read_text(encoding="utf-8")
    public_md = (report_dir / phase53.MD_REPORT.name).read_text(encoding="utf-8")

    assert report["results"][0]["file_id"] == "file_001"
    assert report["results"][0]["filename_hash"]
    assert raw_filename not in public_json
    assert raw_filename not in public_md
    assert extracted_text not in public_json
    assert "Jane Doe" not in public_json
    assert "MRN ABC123" not in public_md


def test_phase56_private_mapping_remains_ignored_by_git():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "check-ignore", "reports/phase53_blind_generalization_audit/local_filename_mapping_PRIVATE.json"],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0


def test_phase56_ui_mentions_supported_image_formats():
    source = (Path(__file__).resolve().parents[1] / "app" / "main.py").read_text(encoding="utf-8")

    assert "Supported formats: PDF, TXT, TIF, TIFF, PNG, JPG, JPEG, BMP, WEBP." in source


def test_phase56_no_external_api_flags_are_enabled():
    assert phase53.app_config.MEDAI_LOCAL_ONLY is True
    assert phase53.app_config.MEDAI_ALLOW_EXTERNAL_API is False


def test_phase56_existing_pdf_txt_behavior_remains_discoverable(tmp_path: Path):
    input_dir = tmp_path / "real_validation_input"
    input_dir.mkdir()
    (input_dir / "a.pdf").write_bytes(b"%PDF")
    (input_dir / "b.txt").write_text("text", encoding="utf-8")

    files = phase53.supported_input_files(input_dir)

    assert [path.suffix.lower() for path in files] == [".pdf", ".txt"]


def test_phase56_no_images_or_pdfs_are_tracked_under_reports():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "ls-files", "reports"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    forbidden_suffixes = (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
    tracked_forbidden = [line for line in result.stdout.splitlines() if line.lower().endswith(forbidden_suffixes)]

    assert tracked_forbidden == []


def test_phase56_extract_image_text_handles_corrupt_image_safely(tmp_path: Path):
    path = tmp_path / "corrupt.png"
    path.write_bytes(b"not an image")

    result = extract_image_text(path)

    assert result.text == ""
    assert result.error
    assert "image_open_failed" in result.warnings or "tesseract_binary_not_found" in result.warnings
