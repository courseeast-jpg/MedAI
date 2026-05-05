from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts import run_phase53_blind_pdf_generalization_audit as phase53
from scripts import run_phase57_full_corpus_inventory_audit as phase57
from scripts import run_phase64_rtf_local_text_parser as phase64
from text_extraction.rtf_text import SUPPORTED_RTF_EXTENSIONS, extract_rtf_text, rtf_to_text


class FakeRtfPipeline:
    def __init__(self):
        self.router = SimpleNamespace(gemini_quota_blocked=False)

    def run(self, job):
        text = str(job.text or "")
        entities = [{"type": "lab", "value": "glucose"}] if "Glucose" in text else []
        confidence = 0.82 if entities else 0.38
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


def write_rtf(path: Path, body: str) -> None:
    path.write_text(r"{\rtf1\ansi " + body + "}", encoding="utf-8")


def test_phase64_rtf_parser_strips_control_syntax(tmp_path: Path):
    path = tmp_path / "synthetic.rtf"
    write_rtf(path, r"\b Glucose\b0  103 mg/dL\par WBC 6.2 x10E3/uL")

    result = extract_rtf_text(path)

    assert result.error is None
    assert "Glucose" in result.text
    assert "WBC 6.2" in result.text
    assert r"\b" not in result.text
    assert "{\\rtf" not in result.text


def test_phase64_rtf_parser_handles_unicode_and_hex_controls():
    text = rtf_to_text(r"{\rtf1 Unicode \u1043? and hex \'47lucose}")

    assert "Unicode" in text
    assert "Glucose" in text


def test_phase64_supported_extensions_include_rtf_without_losing_existing_formats():
    expected_existing = {".pdf", ".txt", ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    assert SUPPORTED_RTF_EXTENSIONS == {".rtf"}
    assert ".rtf" in phase53.SUPPORTED_EXTENSIONS
    assert expected_existing <= phase53.SUPPORTED_EXTENSIONS
    assert expected_existing <= phase57.SUPPORTED_EXTENSIONS


def test_phase64_phase53_processes_synthetic_rtf_locally(tmp_path: Path):
    input_dir = tmp_path / "real_validation_input"
    report_dir = tmp_path / "reports" / "phase53"
    input_dir.mkdir()
    write_rtf(input_dir / "synthetic-lab.rtf", r"Glucose 103 mg/dL 65-99 H")

    report = phase53.run_audit(input_dir=input_dir, report_dir=report_dir, pipeline=FakeRtfPipeline())
    item = report["results"][0]

    assert report["total_files"] == 1
    assert report["external_api_used"] is False
    assert item["file_type"] == "rtf_text"
    assert item["selected_ocr_engine"] == "local_rtf_text_parser"
    assert item["document_type"] == "rtf_text"
    assert item["status"] in {"accepted", "review", "review_ocr_quality"}


def test_phase64_public_phase53_reports_exclude_raw_rtf_filename_and_text(tmp_path: Path):
    input_dir = tmp_path / "real_validation_input"
    report_dir = tmp_path / "reports" / "phase53"
    input_dir.mkdir()
    raw_filename = "Patient Jane Doe DOB 1970 note.rtf"
    raw_text = "Patient Jane Doe DOB 01/02/1970 MRN ABC123 Glucose 103"
    write_rtf(input_dir / raw_filename, raw_text)

    report = phase53.run_audit(input_dir=input_dir, report_dir=report_dir, pipeline=FakeRtfPipeline())
    public_json = (report_dir / phase53.JSON_REPORT.name).read_text(encoding="utf-8")
    public_md = (report_dir / phase53.MD_REPORT.name).read_text(encoding="utf-8")

    assert report["results"][0]["filename_hash"]
    assert raw_filename not in public_json
    assert raw_filename not in public_md
    assert raw_text not in public_json
    assert "Jane Doe" not in public_json
    assert "MRN ABC123" not in public_md


def test_phase64_corrupt_or_empty_rtf_does_not_auto_accept(tmp_path: Path):
    input_dir = tmp_path / "real_validation_input"
    report_dir = tmp_path / "reports" / "phase53"
    input_dir.mkdir()
    (input_dir / "empty.rtf").write_text(r"{\rtf1\ansi }", encoding="utf-8")

    report = phase53.run_audit(input_dir=input_dir, report_dir=report_dir, pipeline=FakeRtfPipeline())
    item = report["results"][0]

    assert item["status"] in {"review", "review_ocr_quality", "error"}
    assert item["status"] != "accepted"
    assert report["external_api_used"] is False


def test_phase64_report_summarizes_rtf_without_private_values(tmp_path: Path):
    phase57_report = tmp_path / "phase57.json"
    phase63_report = tmp_path / "phase63.json"
    private_mapping = tmp_path / "local_filename_mapping_PRIVATE.json"
    report_dir = tmp_path / "reports" / "phase64"
    phase57_payload = {
        "external_api_used": False,
        "raw_phi_logged_in_public_reports": False,
        "unsupported_extension_distribution": {".docx": 1, ".msg": 1, ".mp3": 1, ".ogg": 1},
        "filesystem_reconciliation": {"reconciliation_passed": True},
        "results": [
            {
                "safe_file_id": "corpus_file_000001",
                "filename_hash": "hash_rtf",
                "content_hash": "content_rtf",
                "extension": ".rtf",
                "file_type": "rtf_text",
                "status": "review",
                "selected_extractor": "spacy",
                "confidence": 0.42,
                "classification_reason_codes": ["extraction_low_confidence"],
                "review_reason_codes": ["extraction_low_confidence"],
                "accounting_category": "supported_processed",
                "external_api_used": False,
            }
        ],
    }
    phase57_report.write_text(json.dumps(phase57_payload), encoding="utf-8")
    phase63_report.write_text(json.dumps({"recommended_action_by_extension": {".rtf": {"classification": "safe_to_support_now"}}}), encoding="utf-8")
    private_mapping.write_text(
        json.dumps(
            {
                "files": {
                    "corpus_file_000001": {
                        "original_filename": "Patient Jane Doe note.rtf",
                        "original_relative_path": "Secret Folder/Patient Jane Doe note.rtf",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    report = phase64.run_phase64_report(
        phase57_report_path=phase57_report,
        phase63_report_path=phase63_report,
        private_mapping_path=private_mapping,
        report_dir=report_dir,
    )
    public_json = (report_dir / phase64.JSON_REPORT.name).read_text(encoding="utf-8")
    public_md = (report_dir / phase64.MD_REPORT.name).read_text(encoding="utf-8")

    assert report["rtf_file_count"] == 1
    assert report["rtf_counts"]["supported_processed"] == 1
    assert report["non_rtf_extensions_left_unsupported"] == {".docx": 1, ".mp3": 1, ".msg": 1, ".ogg": 1}
    assert report["external_api_used"] is False
    assert report["raw_phi_logged_in_public_reports"] is False
    assert "Patient Jane Doe" not in public_json
    assert "Secret Folder" not in public_md
