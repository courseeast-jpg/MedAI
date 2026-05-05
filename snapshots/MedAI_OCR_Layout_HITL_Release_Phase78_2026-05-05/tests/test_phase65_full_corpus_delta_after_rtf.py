from __future__ import annotations

import json
from pathlib import Path

from scripts import run_phase65_full_corpus_delta_after_rtf as phase65


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_phase65_computes_before_after_unsupported_delta(tmp_path: Path):
    phase57 = tmp_path / "phase57.json"
    phase63 = tmp_path / "phase63.json"
    phase64 = tmp_path / "phase64.json"
    private = tmp_path / "local_filename_mapping_PRIVATE.json"
    report_dir = tmp_path / "reports"
    write_json(
        phase57,
        {
            "unsupported_count": 8,
            "unsupported_extension_distribution": {".docx": 3, ".mp3": 3, ".msg": 1, ".ogg": 1},
            "total_supported": 606,
            "total_processed": 606,
            "accepted": 93,
            "review": 513,
            "review_ocr_quality": 16,
            "empty": 382,
            "errors": 8,
            "external_api_used": False,
            "raw_phi_logged_in_public_reports": False,
            "filesystem_reconciliation": {"reconciliation_passed": True},
        },
    )
    write_json(phase63, {"extension_distribution": {".docx": 3, ".mp3": 3, ".msg": 1, ".ogg": 1}})
    write_json(
        phase64,
        {
            "rtf_file_count": 3,
            "rtf_counts": {
                "total": 3,
                "supported_processed": 3,
                "unsupported_extension": 0,
                "accepted": 0,
                "review": 3,
                "review_ocr_quality": 0,
                "empty": 0,
                "errors": 0,
            },
            "rtf_moved_from_unsupported_to_supported_or_safe_error": True,
            "external_api_used": False,
            "raw_phi_logged_in_public_reports": False,
            "reconciliation_passed": True,
            "rtf_files": [],
        },
    )
    write_json(private, {"files": {}})

    report = phase65.run_delta_report(
        phase57_report_path=phase57,
        phase63_report_path=phase63,
        phase64_report_path=phase64,
        private_mapping_path=private,
        report_dir=report_dir,
    )

    assert report["before"]["unsupported_extension_count"] == 11
    assert report["after"]["unsupported_extension_count"] == 8
    assert report["delta"]["unsupported_extension_count"] == -3
    assert report["before"]["unsupported_extension_distribution"][".rtf"] == 3
    assert report["rtf_moved_from_unsupported_to_supported_processed"] is True


def test_phase65_confirms_non_rtf_formats_remain_unsupported(tmp_path: Path):
    phase57, phase63, phase64, private, report_dir = write_minimal_inputs(tmp_path)

    report = phase65.run_delta_report(
        phase57_report_path=phase57,
        phase63_report_path=phase63,
        phase64_report_path=phase64,
        private_mapping_path=private,
        report_dir=report_dir,
    )

    assert report["non_rtf_extensions_left_unsupported"] == {".docx": 3, ".mp3": 3, ".msg": 1, ".ogg": 1}
    assert report["docx_mp3_msg_ogg_remain_unsupported_or_excluded"] is True


def test_phase65_status_delta_comes_from_rtf_records_only(tmp_path: Path):
    phase57, phase63, phase64, private, report_dir = write_minimal_inputs(tmp_path)

    report = phase65.run_delta_report(
        phase57_report_path=phase57,
        phase63_report_path=phase63,
        phase64_report_path=phase64,
        private_mapping_path=private,
        report_dir=report_dir,
    )

    assert report["delta"]["accepted"] == 0
    assert report["delta"]["review"] == 3
    assert report["delta"]["empty"] == 0
    assert report["delta"]["errors"] == -3
    assert report["production_extractor_should_change_yet"] is False


def test_phase65_safety_delta_and_conclusion(tmp_path: Path):
    phase57, phase63, phase64, private, report_dir = write_minimal_inputs(tmp_path)

    report = phase65.run_delta_report(
        phase57_report_path=phase57,
        phase63_report_path=phase63,
        phase64_report_path=phase64,
        private_mapping_path=private,
        report_dir=report_dir,
    )

    assert report["external_api_used"] is False
    assert report["raw_phi_logged"] is False
    assert report["safety_regression"] is False
    assert report["reconciliation_passed"] is True
    assert report["conclusion"] == "rtf_delta_measured_no_safety_regression"


def test_phase65_public_reports_use_safe_ids_only(tmp_path: Path):
    phase57, phase63, phase64, private, report_dir = write_minimal_inputs(tmp_path)
    write_json(
        private,
        {
            "files": {
                "corpus_file_000001": {
                    "original_filename": "Patient Jane Doe note.rtf",
                    "original_relative_path": "Private Folder/Patient Jane Doe note.rtf",
                }
            }
        },
    )

    report = phase65.run_delta_report(
        phase57_report_path=phase57,
        phase63_report_path=phase63,
        phase64_report_path=phase64,
        private_mapping_path=private,
        report_dir=report_dir,
    )
    public_json = (report_dir / phase65.JSON_REPORT.name).read_text(encoding="utf-8")
    public_md = (report_dir / phase65.MD_REPORT.name).read_text(encoding="utf-8")

    assert "corpus_file_000001" in public_json
    assert "hash_rtf" in public_md
    assert "Patient Jane Doe" not in public_json
    assert "Private Folder" not in public_md
    assert report["raw_phi_logged"] is False


def test_phase65_blocks_if_private_values_leak(tmp_path: Path, monkeypatch):
    phase57, phase63, phase64, private, report_dir = write_minimal_inputs(tmp_path)
    monkeypatch.setattr(phase65, "render_markdown", lambda report: "Patient Jane Doe leaked")
    write_json(
        private,
        {
            "files": {
                "corpus_file_000001": {
                    "original_filename": "Patient Jane Doe leaked",
                    "original_relative_path": "Private Folder/Patient Jane Doe leaked",
                }
            }
        },
    )

    report = phase65.run_delta_report(
        phase57_report_path=phase57,
        phase63_report_path=phase63,
        phase64_report_path=phase64,
        private_mapping_path=private,
        report_dir=report_dir,
    )

    assert report["raw_phi_logged"] is True
    assert report["conclusion"] == "BLOCKED_PRIVACY_RISK"


def test_phase65_forces_local_only_defaults(tmp_path: Path):
    phase57, phase63, phase64, private, report_dir = write_minimal_inputs(tmp_path)

    report = phase65.run_delta_report(
        phase57_report_path=phase57,
        phase63_report_path=phase63,
        phase64_report_path=phase64,
        private_mapping_path=private,
        report_dir=report_dir,
    )

    assert report["local_only_forced"] is True
    assert phase65.app_config.MEDAI_LOCAL_ONLY is True
    assert phase65.app_config.MEDAI_ALLOW_EXTERNAL_API is False


def test_phase65_recommends_next_problem_class_not_new_format_by_default(tmp_path: Path):
    phase57, phase63, phase64, private, report_dir = write_minimal_inputs(tmp_path)

    report = phase65.run_delta_report(
        phase57_report_path=phase57,
        phase63_report_path=phase63,
        phase64_report_path=phase64,
        private_mapping_path=private,
        report_dir=report_dir,
    )

    assert "DOCX" in report["recommendation"]["docx"]
    assert report["recommendation"]["recommended_next_class"] == "pdf_ocr_low_quality"
    assert report["recommendation"]["next_problem_class_options"] == ["pdf_ocr_low_quality", "image_ocr_low_quality"]


def write_minimal_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    phase57 = tmp_path / "phase57.json"
    phase63 = tmp_path / "phase63.json"
    phase64 = tmp_path / "phase64.json"
    private = tmp_path / "local_filename_mapping_PRIVATE.json"
    report_dir = tmp_path / "reports"
    write_json(
        phase57,
        {
            "unsupported_count": 8,
            "unsupported_extension_distribution": {".docx": 3, ".mp3": 3, ".msg": 1, ".ogg": 1},
            "total_supported": 606,
            "total_processed": 606,
            "accepted": 93,
            "review": 513,
            "review_ocr_quality": 16,
            "empty": 382,
            "errors": 8,
            "external_api_used": False,
            "raw_phi_logged_in_public_reports": False,
            "filesystem_reconciliation": {"reconciliation_passed": True},
        },
    )
    write_json(phase63, {"extension_distribution": {".docx": 3, ".mp3": 3, ".msg": 1, ".ogg": 1}})
    write_json(
        phase64,
        {
            "rtf_file_count": 3,
            "rtf_counts": {
                "total": 3,
                "supported_processed": 3,
                "unsupported_extension": 0,
                "accepted": 0,
                "review": 3,
                "review_ocr_quality": 0,
                "empty": 0,
                "errors": 0,
            },
            "rtf_moved_from_unsupported_to_supported_or_safe_error": True,
            "external_api_used": False,
            "raw_phi_logged_in_public_reports": False,
            "reconciliation_passed": True,
            "rtf_files": [
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
        },
    )
    write_json(private, {"files": {}})
    return phase57, phase63, phase64, private, report_dir
