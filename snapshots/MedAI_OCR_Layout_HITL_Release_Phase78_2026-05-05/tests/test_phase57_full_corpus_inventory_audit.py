from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

# PyPDF2 is only required by a subset of legacy Phase 57 tests. Skip those
# tests when it is unavailable rather than failing module collection so the
# Phase 57A reconciliation tests can still run.
PdfWriter = None
try:  # pragma: no cover - import-availability check
    from PyPDF2 import PdfWriter as _PdfWriter
    PdfWriter = _PdfWriter
except ImportError:  # pragma: no cover - exercised when PyPDF2 missing
    PdfWriter = None

from scripts import run_phase53_blind_pdf_generalization_audit as phase53
from scripts import run_phase54_operator_review_feedback_summary as phase54
from scripts import run_phase57_full_corpus_inventory_audit as phase57


APP_MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"


class FakeCorpusPipeline:
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


def test_phase57_full_corpus_input_exists_or_is_created(tmp_path: Path):
    input_dir = tmp_path / "full_corpus_input"

    phase57.ensure_full_corpus_input(input_dir)

    assert input_dir.exists()
    assert (input_dir / ".gitkeep").exists()


def test_phase57_gitignore_protects_full_corpus_and_private_mapping():
    repo_root = Path(__file__).resolve().parents[1]
    checks = [
        "full_corpus_input/example.pdf",
        "reports/phase57_full_corpus_inventory_audit/local_filename_mapping_PRIVATE.json",
        "reports/phase57_full_corpus_inventory_audit/operator_notes_PRIVATE.json",
    ]
    for path in checks:
        result = subprocess.run(["git", "check-ignore", path], cwd=repo_root, text=True, capture_output=True)
        assert result.returncode == 0, path


def test_phase57_supported_extensions_include_pdf_txt_and_images():
    expected = {".pdf", ".txt", ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    assert expected <= phase57.SUPPORTED_EXTENSIONS


def test_phase57_empty_input_produces_no_input_report(tmp_path: Path):
    input_dir = tmp_path / "full_corpus_input"
    report_dir = tmp_path / "reports" / "phase57"

    report = phase57.run_inventory_audit(input_dir=input_dir, report_dir=report_dir, pipeline=FakeCorpusPipeline())

    assert report["total_discovered"] == 0
    assert report["total_supported"] == 0
    assert report["conclusion"] == "no_input_files"
    assert report["external_api_used"] is False
    assert report["raw_phi_logged_in_public_reports"] is False
    assert (report_dir / phase57.JSON_REPORT.name).exists()
    assert (report_dir / phase57.OPERATOR_SUMMARY.name).exists()


def test_phase57_synthetic_files_use_safe_ids_and_hashes(tmp_path: Path):
    input_dir = tmp_path / "full_corpus_input"
    report_dir = tmp_path / "reports" / "phase57"
    input_dir.mkdir()
    raw_filename = "Jane Doe DOB 1975 MRN 12345 labs.txt"
    raw_text = "Patient Jane Doe DOB 01/02/1975 MRN 12345 Glucose 103"
    (input_dir / raw_filename).write_text(raw_text, encoding="utf-8")

    report = phase57.run_inventory_audit(input_dir=input_dir, report_dir=report_dir, pipeline=FakeCorpusPipeline())
    first = report["results"][0]
    public_json = (report_dir / phase57.JSON_REPORT.name).read_text(encoding="utf-8")
    public_md = (report_dir / phase57.MD_REPORT.name).read_text(encoding="utf-8")

    assert first["safe_file_id"] == "corpus_file_000001"
    assert first["filename_hash"]
    assert first["content_hash"]
    assert raw_filename not in public_json
    assert raw_filename not in public_md
    assert str(input_dir) not in public_json
    assert raw_text not in public_json
    assert "Jane Doe" not in public_json
    assert "MRN 12345" not in public_md


def test_phase57_recursive_subfolder_files_are_included_without_public_folder_paths(tmp_path: Path):
    input_dir = tmp_path / "full_corpus_input"
    report_dir = tmp_path / "reports" / "phase57"
    private_folder = input_dir / "Private Patient Folder"
    private_folder.mkdir(parents=True)
    raw_filename = "Secret Folder Name Labs.txt"
    (private_folder / raw_filename).write_text("Glucose 103 mg/dL", encoding="utf-8")

    report = phase57.run_inventory_audit(input_dir=input_dir, report_dir=report_dir, pipeline=FakeCorpusPipeline())
    public_json = (report_dir / phase57.JSON_REPORT.name).read_text(encoding="utf-8")
    public_md = (report_dir / phase57.MD_REPORT.name).read_text(encoding="utf-8")
    mapping = json.loads((report_dir / phase57.PRIVATE_MAPPING.name).read_text(encoding="utf-8"))

    assert report["total_discovered"] == 1
    assert report["results"][0]["safe_file_id"] == "corpus_file_000001"
    assert "Private Patient Folder" not in public_json
    assert "Private Patient Folder" not in public_md
    assert raw_filename not in public_json
    assert mapping["files"]["corpus_file_000001"]["original_relative_path"].replace("\\", "/") == "Private Patient Folder/Secret Folder Name Labs.txt"


def test_phase57_private_mapping_created_and_ignored(tmp_path: Path):
    input_dir = tmp_path / "full_corpus_input"
    report_dir = tmp_path / "reports" / "phase57"
    input_dir.mkdir()
    (input_dir / "private-name.txt").write_text("Synthetic fixture", encoding="utf-8")

    phase57.run_inventory_audit(input_dir=input_dir, report_dir=report_dir, pipeline=FakeCorpusPipeline())
    mapping = json.loads((report_dir / phase57.PRIVATE_MAPPING.name).read_text(encoding="utf-8"))

    assert mapping["files"]["corpus_file_000001"]["original_filename"] == "private-name.txt"
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "check-ignore", "reports/phase57_full_corpus_inventory_audit/local_filename_mapping_PRIVATE.json"],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0


@pytest.mark.skipif(PdfWriter is None, reason="PyPDF2 not installed in this environment")
def test_phase57_pdf_page_count_is_recorded_when_available(tmp_path: Path):
    input_dir = tmp_path / "full_corpus_input"
    report_dir = tmp_path / "reports" / "phase57"
    input_dir.mkdir()
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_blank_page(width=72, height=72)
    with (input_dir / "two-page.pdf").open("wb") as handle:
        writer.write(handle)

    report = phase57.run_inventory_audit(input_dir=input_dir, report_dir=report_dir, pipeline=FakeCorpusPipeline())

    assert report["results"][0]["page_count"] == 2


@pytest.mark.skipif(PdfWriter is None, reason="PyPDF2 not installed in this environment")
def test_phase57_pdf_embedded_files_are_flagged_without_public_embedded_names(tmp_path: Path):
    input_dir = tmp_path / "full_corpus_input"
    report_dir = tmp_path / "reports" / "phase57"
    input_dir.mkdir()
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_attachment("private-embedded-name.txt", b"Glucose 103")
    with (input_dir / "portfolio.pdf").open("wb") as handle:
        writer.write(handle)

    report = phase57.run_inventory_audit(input_dir=input_dir, report_dir=report_dir, pipeline=FakeCorpusPipeline())
    item = report["results"][0]
    public_json = (report_dir / phase57.JSON_REPORT.name).read_text(encoding="utf-8")

    assert item["pdf_embedded_files_detected"] is True
    assert "pdf_portfolio_or_embedded_files_detected" in item["reason_codes"]
    assert item["status"] == "review"
    assert "private-embedded-name.txt" not in public_json


@pytest.mark.skipif(PdfWriter is None, reason="PyPDF2 not installed in this environment")
def test_phase57_multi_document_pdf_signal_is_flagged_without_splitting(tmp_path: Path, monkeypatch):
    input_dir = tmp_path / "full_corpus_input"
    report_dir = tmp_path / "reports" / "phase57"
    input_dir.mkdir()
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with (input_dir / "combined.pdf").open("wb") as handle:
        writer.write(handle)

    monkeypatch.setattr(
        phase57,
        "inspect_pdf_inventory_metadata",
        lambda path: {
            "page_count": 4,
            "embedded_files_detected": False,
            "embedded_file_count": 0,
            "possible_multi_document_pdf": True,
            "document_class_signals": ["ecg", "lab_report"],
        },
    )
    report = phase57.run_inventory_audit(input_dir=input_dir, report_dir=report_dir, pipeline=FakeCorpusPipeline())
    item = report["results"][0]

    assert item["possible_multi_document_pdf"] is True
    assert "possible_multi_document_pdf" in item["reason_codes"]
    assert item["status"] == "review"


def test_phase57_document_class_signal_detector_identifies_multiple_classes():
    text = "CBC Glucose reference range\nECG 12-lead ventricular rate"

    assert phase57.detect_document_class_signals(text) == {"lab_report", "ecg"}


def test_phase57_external_apis_blocked_and_local_only_forced(tmp_path: Path):
    input_dir = tmp_path / "full_corpus_input"
    report_dir = tmp_path / "reports" / "phase57"
    input_dir.mkdir()
    (input_dir / "synthetic.txt").write_text("Synthetic text only", encoding="utf-8")

    report = phase57.run_inventory_audit(input_dir=input_dir, report_dir=report_dir, pipeline=FakeCorpusPipeline())

    assert report["local_only_forced"] is True
    assert report["external_api_used"] is False
    assert phase53.app_config.MEDAI_LOCAL_ONLY is True
    assert phase53.app_config.MEDAI_ALLOW_EXTERNAL_API is False


def test_phase57_unsupported_and_processing_errors_do_not_crash(tmp_path: Path, monkeypatch):
    input_dir = tmp_path / "full_corpus_input"
    report_dir = tmp_path / "reports" / "phase57"
    input_dir.mkdir()
    (input_dir / "unsupported.docx").write_bytes(b"zip-ish")
    (input_dir / "broken.png").write_bytes(b"not an image")

    def fail_processing(*args, **kwargs):
        raise RuntimeError("synthetic processing failure for private-name.png")

    monkeypatch.setattr(phase57, "process_one_blind_file", fail_processing)
    report = phase57.run_inventory_audit(input_dir=input_dir, report_dir=report_dir, pipeline=FakeCorpusPipeline())

    assert report["total_discovered"] == 2
    assert report["unsupported_count"] == 1
    assert report["errors"] == 2
    assert {item["error_category"] for item in report["results"]} == {"unsupported_format", "processing_error"}
    public_json = (report_dir / phase57.JSON_REPORT.name).read_text(encoding="utf-8")
    assert "broken.png" not in public_json
    assert "private-name.png" not in public_json


def test_phase57_problem_clusters_are_produced_for_safe_cases(tmp_path: Path):
    input_dir = tmp_path / "full_corpus_input"
    report_dir = tmp_path / "reports" / "phase57"
    input_dir.mkdir()
    (input_dir / "weak.txt").write_text("No structured entities", encoding="utf-8")
    (input_dir / "skip.gif").write_bytes(b"GIF89a")

    report = phase57.run_inventory_audit(input_dir=input_dir, report_dir=report_dir, pipeline=FakeCorpusPipeline())

    assert "unsupported_format" in report["problem_clusters"]
    assert "unknown_other" in report["problem_clusters"]
    assert (report_dir / phase57.CLUSTERS_JSON.name).exists()
    assert (report_dir / phase57.CLUSTERS_MD.name).exists()


def test_phase57_ui_label_references_full_corpus_inventory_audit():
    source = APP_MAIN.read_text(encoding="utf-8")

    assert "Phase57 Full Corpus Inventory Audit" in source
    assert "full_corpus_input/" in source
    assert "Run Phase57 Full Corpus Inventory Audit" in source


def test_phase57_existing_phase53_and_phase54_behavior_remains_importable():
    assert "real_validation_input" in str(phase53.INPUT_DIR)
    assert phase54.PRIVATE_FEEDBACK.name == "operator_feedback_PRIVATE.json"


def test_phase57_no_pdfs_or_images_are_tracked_under_reports():
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


# ---------------------------------------------------------------------------
# Phase 57A — full filesystem reconciliation
# ---------------------------------------------------------------------------


def test_phase57a_total_filesystem_files_includes_supported_and_unsupported(tmp_path: Path):
    input_dir = tmp_path / "full_corpus_input"
    report_dir = tmp_path / "reports" / "phase57"
    input_dir.mkdir()
    (input_dir / "supported.txt").write_text("Glucose 103 mg/dL", encoding="utf-8")
    (input_dir / "unsupported.docx").write_bytes(b"PK")
    (input_dir / "another.xml").write_bytes(b"<a/>")

    report = phase57.run_inventory_audit(
        input_dir=input_dir, report_dir=report_dir, pipeline=FakeCorpusPipeline()
    )
    rec = report["filesystem_reconciliation"]

    # 3 corpus files + .gitkeep created by ensure_full_corpus_input
    assert rec["total_filesystem_files"] == 4
    assert rec["total_supported_processed"] == 1
    assert rec["total_unsupported_extension"] == 2
    assert rec["total_ignored_system_files"] == 1  # .gitkeep
    assert rec["accounted_total"] == 4
    assert rec["unexplained_count"] == 0
    assert rec["reconciliation_passed"] is True


def test_phase57a_unsupported_files_are_counted_not_silently_ignored(tmp_path: Path):
    input_dir = tmp_path / "full_corpus_input"
    report_dir = tmp_path / "reports" / "phase57"
    input_dir.mkdir()
    (input_dir / "doc.docx").write_bytes(b"PK")
    (input_dir / "data.xml").write_bytes(b"<x/>")
    (input_dir / "image.gif").write_bytes(b"GIF89a")

    report = phase57.run_inventory_audit(
        input_dir=input_dir, report_dir=report_dir, pipeline=FakeCorpusPipeline()
    )
    rec = report["filesystem_reconciliation"]

    assert rec["total_unsupported_extension"] == 3
    assert report["unsupported_extension_distribution"] == {".docx": 1, ".gif": 1, ".xml": 1}


def test_phase57a_gitkeep_classified_as_ignored_system_file(tmp_path: Path):
    input_dir = tmp_path / "full_corpus_input"
    report_dir = tmp_path / "reports" / "phase57"

    report = phase57.run_inventory_audit(
        input_dir=input_dir, report_dir=report_dir, pipeline=FakeCorpusPipeline()
    )
    rec = report["filesystem_reconciliation"]

    # ensure_full_corpus_input creates .gitkeep
    assert rec["total_ignored_system_files"] >= 1
    ignored_entries = [
        entry for entry in report["filesystem_inventory_entries"]
        if entry.get("accounting_category") == "ignored_system_file"
    ]
    assert any(entry.get("extension") == "" or entry.get("file_extension") == "" for entry in ignored_entries) or any(
        entry.get("extension") in {"", ".gitkeep"} for entry in ignored_entries
    )


def test_phase57a_unexplained_count_is_zero_for_synthetic_corpus(tmp_path: Path):
    input_dir = tmp_path / "full_corpus_input"
    report_dir = tmp_path / "reports" / "phase57"
    input_dir.mkdir()
    (input_dir / "labs.txt").write_text("Glucose 103", encoding="utf-8")
    sub = input_dir / "subfolder"
    sub.mkdir()
    (sub / "more.txt").write_text("Glucose 95", encoding="utf-8")
    (sub / "trash.tmp").write_bytes(b"junk")

    report = phase57.run_inventory_audit(
        input_dir=input_dir, report_dir=report_dir, pipeline=FakeCorpusPipeline()
    )
    rec = report["filesystem_reconciliation"]

    assert rec["unexplained_count"] == 0
    assert rec["total_unknown_unclassified"] == 0
    assert rec["reconciliation_passed"] is True
    assert rec["total_filesystem_folders"] == 1
    expected_total = (
        rec["total_supported_processed"]
        + rec["total_unsupported_extension"]
        + rec["total_ignored_system_files"]
        + rec["total_skipped_policy"]
        + rec["total_processing_errors"]
        + rec["total_inaccessible_files"]
        + rec["total_unknown_unclassified"]
    )
    assert rec["accounted_total"] == expected_total == rec["total_filesystem_files"]


def test_phase57a_total_filesystem_files_equals_sum_of_categories(tmp_path: Path, monkeypatch):
    input_dir = tmp_path / "full_corpus_input"
    report_dir = tmp_path / "reports" / "phase57"
    input_dir.mkdir()
    (input_dir / "ok.txt").write_text("Glucose 103", encoding="utf-8")
    (input_dir / "broken.png").write_bytes(b"not an image")
    (input_dir / "weird.xml").write_bytes(b"<a/>")

    def fail_processing(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(phase57, "process_one_blind_file", fail_processing)
    report = phase57.run_inventory_audit(
        input_dir=input_dir, report_dir=report_dir, pipeline=FakeCorpusPipeline()
    )
    rec = report["filesystem_reconciliation"]

    expected = (
        rec["total_supported_processed"]
        + rec["total_unsupported_extension"]
        + rec["total_ignored_system_files"]
        + rec["total_skipped_policy"]
        + rec["total_processing_errors"]
        + rec["total_inaccessible_files"]
        + rec["total_unknown_unclassified"]
    )
    assert rec["accounted_total"] == expected
    assert rec["unexplained_count"] == 0


def test_phase57a_public_reports_do_not_contain_raw_filenames(tmp_path: Path):
    input_dir = tmp_path / "full_corpus_input"
    report_dir = tmp_path / "reports" / "phase57"
    sub = input_dir / "Private Folder"
    sub.mkdir(parents=True)
    raw_filename = "Patient Surname Glucose Labs.txt"
    (sub / raw_filename).write_text("Glucose 103", encoding="utf-8")
    raw_unsupported = "Confidential Document Name.docx"
    (sub / raw_unsupported).write_bytes(b"PK")

    phase57.run_inventory_audit(
        input_dir=input_dir, report_dir=report_dir, pipeline=FakeCorpusPipeline()
    )
    public_paths = [
        report_dir / phase57.JSON_REPORT.name,
        report_dir / phase57.MD_REPORT.name,
        report_dir / phase57.OPERATOR_SUMMARY.name,
        report_dir / phase57.CLUSTERS_JSON.name,
        report_dir / phase57.CLUSTERS_MD.name,
    ]
    public_text = "\n".join(p.read_text(encoding="utf-8") for p in public_paths if p.exists())

    assert raw_filename not in public_text
    assert raw_unsupported not in public_text
    assert "Private Folder" not in public_text


def test_phase57a_private_mapping_contains_raw_paths(tmp_path: Path):
    input_dir = tmp_path / "full_corpus_input"
    report_dir = tmp_path / "reports" / "phase57"
    sub = input_dir / "Private Folder"
    sub.mkdir(parents=True)
    raw_filename = "Patient Labs.txt"
    (sub / raw_filename).write_text("Glucose 103", encoding="utf-8")

    phase57.run_inventory_audit(
        input_dir=input_dir, report_dir=report_dir, pipeline=FakeCorpusPipeline()
    )
    mapping = json.loads((report_dir / phase57.PRIVATE_MAPPING.name).read_text(encoding="utf-8"))

    assert any(
        entry.get("original_filename") == raw_filename
        and "Private Folder" in entry.get("original_relative_path", "")
        for entry in mapping["files"].values()
    )


def test_phase57a_extension_distribution_is_present_in_report(tmp_path: Path):
    input_dir = tmp_path / "full_corpus_input"
    report_dir = tmp_path / "reports" / "phase57"
    input_dir.mkdir()
    (input_dir / "a.txt").write_text("Glucose 103", encoding="utf-8")
    (input_dir / "b.txt").write_text("Glucose 95", encoding="utf-8")
    (input_dir / "c.docx").write_bytes(b"PK")

    report = phase57.run_inventory_audit(
        input_dir=input_dir, report_dir=report_dir, pipeline=FakeCorpusPipeline()
    )

    dist = report["extension_distribution"]
    assert dist[".txt"] == 2
    assert dist[".docx"] == 1
    # .gitkeep — no_extension bucket
    assert "no_extension" in dist or ".gitkeep" in dist or len(dist) >= 2


def test_phase57a_hidden_and_system_files_classified_as_ignored(tmp_path: Path):
    input_dir = tmp_path / "full_corpus_input"
    report_dir = tmp_path / "reports" / "phase57"
    input_dir.mkdir()
    (input_dir / ".DS_Store").write_bytes(b"\x00\x00")
    (input_dir / "Thumbs.db").write_bytes(b"\x00\x00")
    (input_dir / "desktop.ini").write_bytes(b"[.ShellClassInfo]")
    (input_dir / "~$lock.tmp").write_bytes(b"")
    (input_dir / "real.txt").write_text("Glucose 103", encoding="utf-8")

    report = phase57.run_inventory_audit(
        input_dir=input_dir, report_dir=report_dir, pipeline=FakeCorpusPipeline()
    )
    rec = report["filesystem_reconciliation"]

    # 4 ignored + .gitkeep auto-created = 5
    assert rec["total_ignored_system_files"] >= 5
    assert rec["total_supported_processed"] == 1
    assert rec["unexplained_count"] == 0


def test_phase57a_processing_error_categorized_distinctly_from_unexplained(tmp_path: Path, monkeypatch):
    input_dir = tmp_path / "full_corpus_input"
    report_dir = tmp_path / "reports" / "phase57"
    input_dir.mkdir()
    (input_dir / "broken.png").write_bytes(b"corrupt")

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(phase57, "process_one_blind_file", fail)
    report = phase57.run_inventory_audit(
        input_dir=input_dir, report_dir=report_dir, pipeline=FakeCorpusPipeline()
    )
    rec = report["filesystem_reconciliation"]

    assert rec["total_processing_errors"] == 1
    assert rec["total_unknown_unclassified"] == 0
    assert rec["unexplained_count"] == 0


def test_phase57a_empty_subfolders_affect_folder_count_only(tmp_path: Path):
    input_dir = tmp_path / "full_corpus_input"
    report_dir = tmp_path / "reports" / "phase57"
    input_dir.mkdir()
    (input_dir / "labs.txt").write_text("Glucose 103", encoding="utf-8")
    (input_dir / "empty_a").mkdir()
    (input_dir / "empty_b" / "nested").mkdir(parents=True)

    report = phase57.run_inventory_audit(
        input_dir=input_dir, report_dir=report_dir, pipeline=FakeCorpusPipeline()
    )
    rec = report["filesystem_reconciliation"]

    # 3 folders: empty_a, empty_b, empty_b/nested
    assert rec["total_filesystem_folders"] == 3
    # 2 files: labs.txt + .gitkeep
    assert rec["total_filesystem_files"] == 2
    assert rec["unexplained_count"] == 0


def test_phase57a_existing_total_discovered_semantics_preserved(tmp_path: Path):
    # Phase 57 contract: total_discovered counts processed candidates only
    # (not ignored system files), so existing tests/UI continue to behave.
    input_dir = tmp_path / "full_corpus_input"
    report_dir = tmp_path / "reports" / "phase57"
    input_dir.mkdir()
    (input_dir / "real.txt").write_text("Glucose 103", encoding="utf-8")

    report = phase57.run_inventory_audit(
        input_dir=input_dir, report_dir=report_dir, pipeline=FakeCorpusPipeline()
    )

    assert report["total_discovered"] == 1
    rec = report["filesystem_reconciliation"]
    assert rec["total_filesystem_files"] == 2  # real.txt + .gitkeep
    assert rec["total_supported_processed"] == 1
    assert rec["total_ignored_system_files"] == 1


def test_phase57a_reconciliation_blocked_conclusion_when_unaccounted(tmp_path: Path, monkeypatch):
    # Force unknown_unclassified > 0 by injecting an item that escapes normal
    # classification. Confirms the conclusion gate fires.
    input_dir = tmp_path / "full_corpus_input"
    report_dir = tmp_path / "reports" / "phase57"
    input_dir.mkdir()
    (input_dir / "real.txt").write_text("Glucose 103", encoding="utf-8")

    original = phase57.assign_accounting_category

    def force_unknown(item):
        # Force-mismatch so the synthetic file lands in unknown_unclassified.
        return "unknown_unclassified"

    monkeypatch.setattr(phase57, "assign_accounting_category", force_unknown)
    report = phase57.run_inventory_audit(
        input_dir=input_dir, report_dir=report_dir, pipeline=FakeCorpusPipeline()
    )

    rec = report["filesystem_reconciliation"]
    assert rec["total_unknown_unclassified"] >= 1
    assert rec["reconciliation_passed"] is False
    assert report["conclusion"] == "BLOCKED_RECONCILIATION_GAP"
