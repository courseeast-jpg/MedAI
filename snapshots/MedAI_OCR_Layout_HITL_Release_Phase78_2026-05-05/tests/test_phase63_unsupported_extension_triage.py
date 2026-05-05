from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts import run_phase53_blind_pdf_generalization_audit as phase53
from scripts import run_phase63_unsupported_extension_triage as phase63


def write_phase57_fixture(path: Path) -> None:
    payload = {
        "results": [
            {
                "safe_file_id": "corpus_file_000001",
                "filename_hash": "hash_pdf",
                "content_hash": "content_pdf",
                "extension": ".pdf",
                "file_type": "pdf",
                "reason_codes": [],
            },
            {
                "safe_file_id": "corpus_file_000002",
                "filename_hash": "hash_rtf",
                "content_hash": "content_rtf",
                "extension": ".rtf",
                "file_type": "unsupported",
                "error_category": "unsupported_format",
                "reason_codes": ["unsupported_format"],
            },
            {
                "safe_file_id": "corpus_file_000003",
                "filename_hash": "hash_docx",
                "content_hash": "content_docx",
                "extension": ".docx",
                "file_type": "unsupported",
                "error_category": "unsupported_format",
                "reason_codes": ["unsupported_format"],
            },
            {
                "safe_file_id": "corpus_file_000004",
                "filename_hash": "hash_mp3",
                "content_hash": "content_mp3",
                "extension": ".mp3",
                "file_type": "unsupported",
                "error_category": "unsupported_format",
                "reason_codes": ["unsupported_format"],
            },
            {
                "safe_file_id": "corpus_file_000005",
                "filename_hash": "hash_msg",
                "content_hash": "content_msg",
                "extension": ".msg",
                "file_type": "unsupported",
                "error_category": "unsupported_format",
                "reason_codes": ["unsupported_format"],
            },
            {
                "safe_file_id": "corpus_file_000006",
                "filename_hash": "hash_bin",
                "content_hash": "content_bin",
                "extension": ".bin",
                "file_type": "unsupported",
                "error_category": "unsupported_format",
                "reason_codes": ["unsupported_format"],
            },
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_private_mapping(path: Path) -> None:
    payload = {
        "files": {
            "corpus_file_000002": {
                "original_filename": "Patient Jane Doe note.rtf",
                "original_relative_path": "Private Folder/Patient Jane Doe note.rtf",
                "filename_hash": "hash_rtf",
                "content_hash": "content_rtf",
                "accounting_category": "unsupported_extension",
            },
            "corpus_file_000003": {
                "original_filename": "Patient Jane Doe note.docx",
                "original_relative_path": "Private Folder/Patient Jane Doe note.docx",
                "filename_hash": "hash_docx",
                "content_hash": "content_docx",
                "accounting_category": "unsupported_extension",
            },
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_phase63_unsupported_extensions_are_grouped_correctly(tmp_path: Path):
    phase57 = tmp_path / "phase57.json"
    private = tmp_path / "local_filename_mapping_PRIVATE.json"
    report_dir = tmp_path / "reports"
    write_phase57_fixture(phase57)
    write_private_mapping(private)

    report = phase63.run_triage(phase57_report_path=phase57, private_mapping_path=private, report_dir=report_dir)

    assert report["unsupported_count"] == 5
    assert report["extension_distribution"] == {".rtf": 1, ".docx": 1, ".mp3": 1, ".msg": 1, ".bin": 1}
    assert report["recommended_action_by_extension"][".rtf"]["safe_file_ids"] == ["corpus_file_000002"]


def test_phase63_public_reports_use_safe_ids_only_and_no_private_names(tmp_path: Path):
    phase57 = tmp_path / "phase57.json"
    private = tmp_path / "local_filename_mapping_PRIVATE.json"
    report_dir = tmp_path / "reports"
    write_phase57_fixture(phase57)
    write_private_mapping(private)

    report = phase63.run_triage(phase57_report_path=phase57, private_mapping_path=private, report_dir=report_dir)
    public_json = (report_dir / phase63.JSON_REPORT.name).read_text(encoding="utf-8")
    public_md = (report_dir / phase63.MD_REPORT.name).read_text(encoding="utf-8")

    assert "corpus_file_000002" in public_json
    assert "hash_rtf" in public_md
    assert "Patient Jane Doe" not in public_json
    assert "Private Folder" not in public_md
    assert "local_filename_mapping_PRIVATE.json" not in public_json
    assert report["raw_phi_logged_in_public_reports"] is False


def test_phase63_private_mapping_remains_ignored():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "check-ignore", "reports/phase57_full_corpus_inventory_audit/local_filename_mapping_PRIVATE.json"],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0


def test_phase63_text_like_extension_is_safe_to_support_now():
    action = phase63.classify_extension(".rtf", [{"safe_file_id": "file_001"}])

    assert action["classification"] == "safe_to_support_now"
    assert action["recommended_action"] == "support_prototype"
    assert action["production_extractor_should_change_yet"] is False


def test_phase63_binary_archive_and_office_classifications():
    assert phase63.classify_extension(".msg", [{"safe_file_id": "file_001"}])["classification"] == "explicit_exclusion"
    assert phase63.classify_extension(".mp3", [{"safe_file_id": "file_001"}])["classification"] == "explicit_exclusion"
    assert phase63.classify_extension(".docx", [{"safe_file_id": "file_001"}])["classification"] == "support_later"
    assert phase63.classify_extension(".bin", [{"safe_file_id": "file_001"}])["classification"] == "manual_review_only"


def test_phase63_external_apis_remain_disabled(tmp_path: Path):
    phase57 = tmp_path / "phase57.json"
    private = tmp_path / "local_filename_mapping_PRIVATE.json"
    report_dir = tmp_path / "reports"
    write_phase57_fixture(phase57)
    write_private_mapping(private)

    report = phase63.run_triage(phase57_report_path=phase57, private_mapping_path=private, report_dir=report_dir)

    assert report["local_only_forced"] is True
    assert report["external_api_used"] is False
    assert phase63.app_config.MEDAI_LOCAL_ONLY is True
    assert phase63.app_config.MEDAI_ALLOW_EXTERNAL_API is False


def test_phase63_existing_supported_formats_remain_unchanged():
    expected = {".pdf", ".txt", ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    assert expected <= phase53.SUPPORTED_EXTENSIONS
    assert expected <= phase63.SUPPORTED_EXTENSIONS

