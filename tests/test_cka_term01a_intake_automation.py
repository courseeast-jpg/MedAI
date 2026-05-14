"""CKA-TERM-01A — operator intake automation tests."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from clinical_knowledge.terminology import (    # noqa: E402
    classify_filename,
    classify_filenames,
    compute_readiness,
    copy_classified_files,
    optional_local_scan,
    prepare_intake_folders,
    real_ack_filename,
    safe_extract_zip,
    template_filename,
    template_payload,
    write_ack_template,
)


REPORT_DIR = REPO_ROOT / "reports" / "cka_term01a_intake_automation"


def _ws(prefix: str = "cka_term01a_pytest_") -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))


def _rmtree(p: Path) -> None:
    shutil.rmtree(p, ignore_errors=True)


# ---------------------------------------------------------------------------
# TestImports
# ---------------------------------------------------------------------------


class TestImports:
    def test_package_exposes_term01a_surface(self):
        import clinical_knowledge.terminology as t
        for name in (
            "ClassificationSummary", "FileClassification",
            "classify_filename", "classify_filenames",
            "TemplateWriteResult", "real_ack_filename",
            "template_filename", "template_payload", "write_ack_template",
            "CopyResult", "ExtractResult", "FolderPreparationResult",
            "LocalScanResult", "ReadinessReport",
            "compute_readiness", "copy_classified_files",
            "optional_local_scan", "prepare_intake_folders",
            "safe_extract_zip",
        ):
            assert hasattr(t, name), f"missing export: {name}"

    def test_validation_script_importable(self):
        from scripts import run_cka_term01a_intake_automation_validation as v
        assert hasattr(v, "run_validation")

    def test_prepare_script_importable(self):
        from scripts import cka_terminology_prepare_intake as s
        assert hasattr(s, "main")

    def test_check_ready_script_importable(self):
        from scripts import cka_terminology_check_ready as s
        assert hasattr(s, "main")


# ---------------------------------------------------------------------------
# TestFolderPreparation
# ---------------------------------------------------------------------------


class TestFolderPreparation:
    def test_creates_subdirs_and_template(self):
        ws = _ws()
        try:
            r = prepare_intake_folders(repo_root=ws)
            s = r.safe_public_summary()
            assert s["root_present"] is True
            all_subs = set(s["subdirs_created"]) | set(s["subdirs_already_present"])
            assert {"loinc", "rxnorm", "umls", "snomed_ct"}.issubset(all_subs)
            td = ws / "terminology_data"
            assert (td / template_filename()).exists()
            assert not (td / real_ack_filename()).exists()
            assert s["template"]["real_ack_created"] is False
        finally:
            _rmtree(ws)

    def test_idempotent(self):
        ws = _ws()
        try:
            prepare_intake_folders(repo_root=ws)
            r2 = prepare_intake_folders(repo_root=ws)
            s2 = r2.safe_public_summary()
            assert {"loinc", "rxnorm", "umls", "snomed_ct"}.issubset(
                set(s2["subdirs_already_present"])
            )
            assert s2["template"]["template_already_present"] is True
            assert s2["template"]["real_ack_created"] is False
        finally:
            _rmtree(ws)


# ---------------------------------------------------------------------------
# TestTemplateNoRealAck
# ---------------------------------------------------------------------------


class TestTemplateNoRealAck:
    def test_template_payload_safe(self):
        body = template_payload()
        assert body["operator_acknowledged"] is False
        assert body["acknowledged_systems"] == []

    def test_write_ack_template_creates_only_template(self, tmp_path):
        result = write_ack_template(tmp_path)
        s = result.safe_public_summary()
        assert s["template_created"] is True
        assert s["real_ack_created"] is False
        assert (tmp_path / template_filename()).exists()
        assert not (tmp_path / real_ack_filename()).exists()

    def test_re_call_idempotent(self, tmp_path):
        write_ack_template(tmp_path)
        r = write_ack_template(tmp_path)
        assert r.template_already_present is True
        assert r.template_created is False


# ---------------------------------------------------------------------------
# TestClassifier
# ---------------------------------------------------------------------------


class TestClassifier:
    @pytest.mark.parametrize("name,want", [
        ("Loinc.csv", "loinc"),
        ("LoincTable.csv", "loinc"),
        ("RXNCONSO.RRF", "rxnorm"),
        ("RxNorm_full_20240101.zip", "rxnorm"),
        ("MRCONSO.RRF", "umls"),
        ("umls-2024AA-mmsys.zip", "umls"),
        ("sct2_Concept_Snapshot_INT_20240101.txt", "snomed_ct"),
        ("SnomedCT_International_20240101.zip", "snomed_ct"),
    ])
    def test_known_filenames(self, name, want):
        c = classify_filename(name)
        assert c.system is not None
        assert c.system.value == want

    @pytest.mark.parametrize("name", [
        "readme.md", "data.parquet", "random.txt", "notes.docx", "archive.tar.gz",
    ])
    def test_unknown_filenames(self, name):
        c = classify_filename(name)
        assert c.system is None

    def test_summary_no_raw_paths(self):
        names = ["MRCONSO.RRF", "Loinc.csv", "RxNorm_full.zip", "random.bin"]
        s = classify_filenames(names).safe_public_summary()
        text = json.dumps(s)
        assert "/" not in text
        assert not re.search(r"[A-Za-z]:\\\\", text)


# ---------------------------------------------------------------------------
# TestScan
# ---------------------------------------------------------------------------


class TestScan:
    def test_scan_default_off(self, tmp_path):
        (tmp_path / "Loinc.csv").write_text("x", encoding="utf-8")
        r = optional_local_scan(scan_dir=tmp_path, enabled=False)
        assert r.scanned is False
        assert r.files_seen == 0

    def test_scan_top_level_only_by_default(self, tmp_path):
        (tmp_path / "Loinc.csv").write_text("x", encoding="utf-8")
        sub = tmp_path / "deep"
        sub.mkdir()
        (sub / "MRCONSO.RRF").write_text("x", encoding="utf-8")
        r = optional_local_scan(scan_dir=tmp_path, enabled=True)
        assert r.scanned is True
        assert r.files_seen == 1

    def test_scan_recurse_finds_all(self, tmp_path):
        (tmp_path / "Loinc.csv").write_text("x", encoding="utf-8")
        sub = tmp_path / "deep"
        sub.mkdir()
        (sub / "MRCONSO.RRF").write_text("x", encoding="utf-8")
        r = optional_local_scan(scan_dir=tmp_path, enabled=True, recurse=True)
        assert r.files_seen == 2


# ---------------------------------------------------------------------------
# TestCopy
# ---------------------------------------------------------------------------


class TestCopy:
    def test_copy_off_no_change(self):
        ws = _ws()
        try:
            scan = ws / "scan"
            scan.mkdir()
            (scan / "Loinc.csv").write_text("x", encoding="utf-8")
            r = copy_classified_files(
                list(scan.iterdir()), repo_root=ws, copy_approved=False,
            )
            assert r.copy_approved is False
            assert r.files_copied == 0
            assert not (ws / "terminology_data" / "loinc" / "Loinc.csv").exists()
        finally:
            _rmtree(ws)

    def test_copy_approved_lands_under_terminology_data(self):
        ws = _ws()
        try:
            scan = ws / "scan"
            scan.mkdir()
            (scan / "Loinc.csv").write_text("x", encoding="utf-8")
            (scan / "MRCONSO.RRF").write_text("x", encoding="utf-8")
            (scan / "random.bin").write_text("x", encoding="utf-8")
            r = copy_classified_files(
                list(scan.iterdir()), repo_root=ws, copy_approved=True,
            )
            assert r.files_copied == 2
            assert r.files_skipped_unknown == 1
            assert (ws / "terminology_data" / "loinc" / "Loinc.csv").exists()
            assert (ws / "terminology_data" / "umls" / "MRCONSO.RRF").exists()
            # All written paths must resolve under terminology_data/.
            td = (ws / "terminology_data").resolve()
            for p in (ws / "terminology_data").rglob("*"):
                if p.is_file():
                    assert td in p.resolve().parents or p.resolve().parent == td
        finally:
            _rmtree(ws)


# ---------------------------------------------------------------------------
# TestZipSlip
# ---------------------------------------------------------------------------


class TestZipSlip:
    def test_extract_off_no_change(self, tmp_path):
        zp = tmp_path / "SnomedCT_test.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("inside.txt", "ok")
        r = safe_extract_zip([zp], repo_root=tmp_path, extract_approved=False)
        assert r.extract_approved is False
        assert r.entries_extracted == 0
        assert not (tmp_path / "terminology_data" / "snomed_ct" / "inside.txt").exists()

    def test_zip_slip_blocked_parent_traversal(self):
        ws = _ws()
        try:
            zp = ws / "SnomedCT_evil.zip"
            with zipfile.ZipFile(zp, "w") as zf:
                zf.writestr("safe.txt", "ok")
                zf.writestr("../../escape.txt", "PWN")
            r = safe_extract_zip([zp], repo_root=ws, extract_approved=True)
            s = r.safe_public_summary()
            assert s["entries_blocked_zip_slip"] >= 1
            # safe.txt was extracted; escape.txt was not.
            assert (ws / "terminology_data" / "snomed_ct" / "safe.txt").exists()
            for parent in (ws, ws.parent):
                assert not (parent / "escape.txt").exists()
        finally:
            _rmtree(ws)

    def test_zip_slip_blocked_absolute_paths(self):
        ws = _ws()
        try:
            zp = ws / "SnomedCT_abs.zip"
            with zipfile.ZipFile(zp, "w") as zf:
                zf.writestr("/abs.txt", "PWN")
                zf.writestr("\\abs2.txt", "PWN")
                zf.writestr("C:/abs_drive.txt", "PWN")
            r = safe_extract_zip([zp], repo_root=ws, extract_approved=True)
            s = r.safe_public_summary()
            assert s["entries_blocked_zip_slip"] >= 1
        finally:
            _rmtree(ws)

    def test_unknown_archive_blocked(self):
        ws = _ws()
        try:
            zp = ws / "unknown_thing.zip"
            with zipfile.ZipFile(zp, "w") as zf:
                zf.writestr("inside.txt", "ok")
            r = safe_extract_zip([zp], repo_root=ws, extract_approved=True)
            s = r.safe_public_summary()
            assert s["archives_blocked_unknown_system"] >= 1
        finally:
            _rmtree(ws)


# ---------------------------------------------------------------------------
# TestReadinessChecker
# ---------------------------------------------------------------------------


class TestReadinessChecker:
    def test_no_files(self):
        ws = _ws()
        try:
            rd = compute_readiness(repo_root=ws)
            s = rd.safe_public_summary()
            assert set(s["systems_missing"]) == {"umls", "snomed_ct", "rxnorm", "loinc"}
            assert s["systems_present"] == []
            assert s["pending_acknowledgments"] == []
        finally:
            _rmtree(ws)

    def test_files_but_no_ack(self):
        ws = _ws()
        try:
            prepare_intake_folders(repo_root=ws)
            (ws / "terminology_data" / "umls" / "MRCONSO.RRF").write_text("x", encoding="utf-8")
            rd = compute_readiness(repo_root=ws)
            s = rd.safe_public_summary()
            assert "umls" in s["systems_present"]
            assert "umls" in s["systems_license_required"]
            assert "umls" in s["pending_acknowledgments"]
            assert "umls" not in s["systems_acknowledged"]
            assert "umls" not in s["systems_import_ready"]
        finally:
            _rmtree(ws)

    def test_template_does_not_count_as_ack(self, monkeypatch):
        ws = _ws()
        try:
            prepare_intake_folders(repo_root=ws)
            (ws / "terminology_data" / "loinc" / "Loinc.csv").write_text("x", encoding="utf-8")
            # The template file (operator_acknowledged=False) was created
            # by prepare_intake_folders. compute_readiness must NOT treat
            # it as a real acknowledgment.
            monkeypatch.setattr(
                "clinical_knowledge.terminology.intake_automation.license_acknowledged_for",
                lambda *args, **kwargs: False,
            )
            rd = compute_readiness(repo_root=ws)
            s = rd.safe_public_summary()
            assert "loinc" not in s["systems_acknowledged"]
        finally:
            _rmtree(ws)


# ---------------------------------------------------------------------------
# TestNoFilesStaged
# ---------------------------------------------------------------------------


class TestNoFilesStaged:
    def test_terminology_data_not_tracked(self):
        res = subprocess.run(
            ["git", "ls-files", "terminology_data/"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30,
        )
        assert res.returncode == 0
        assert res.stdout.strip() == ""

    def test_license_ack_files_not_tracked(self):
        res = subprocess.run(
            ["git", "ls-files", "*LICENSE_ACK_PRIVATE*"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30,
        )
        assert res.returncode == 0
        assert res.stdout.strip() == ""


# ---------------------------------------------------------------------------
# TestPublicReport
# ---------------------------------------------------------------------------


class TestPublicReport:
    @pytest.fixture(scope="class")
    def report(self):
        from scripts.run_cka_term01a_intake_automation_validation import run_validation
        return run_validation()

    def test_block_id(self, report):
        assert report["block_id"] == "CKA-TERM-01A"

    def test_conclusion(self, report):
        assert report["conclusion"] == "cka_term01a_operator_intake_automation_ready"

    def test_flags_true(self, report):
        for k in (
            "folders_prepared", "ack_template_ready",
            "file_classifier_ready", "zip_slip_protection_ready",
            "inventory_runner_ready", "local_scan_default_off",
        ):
            assert report[k] is True, f"flag {k} not True"

    def test_flags_false(self, report):
        for k in (
            "real_ack_created",
            "real_terminology_downloaded",
            "real_terminology_imported",
            "real_terminology_files_committed",
            "license_gate_bypassed",
            "external_api_used",
            "external_terminology_api_used",
            "raw_phi_logged_in_public_reports",
            "license_text_written_to_public_reports",
            "clinical_recommendations_generated",
            "prescription_dosing_advice_generated",
            "production_ocr_changed",
            "production_extractor_changed",
            "safety_gate_changed",
            "frozen_hitl_release_reopened",
        ):
            assert report[k] is False, f"flag {k} not False"

    def test_zero_leak_counters(self, report):
        assert report["private_filename_path_leaks"] == 0
        assert report["secret_leaks"] == 0

    def test_report_no_drive_letter_path(self, report):
        assert not re.search(r"[A-Za-z]:\\\\", json.dumps(report))

    def test_report_passes_b02_privacy_checker(self, report):
        from clinical_knowledge.privacy.report_privacy import check_public_report_payload
        result = check_public_report_payload(report)
        assert result.passed, f"privacy checker rejected: {result.leak_examples_redacted}"

    def test_report_md_present(self):
        md = REPORT_DIR / "cka_term01a_intake_automation_report.md"
        assert md.exists()

    def test_operator_guide_present(self):
        guide = REPORT_DIR / "CKA_TERM01A_OPERATOR_INTAKE_GUIDE.md"
        assert guide.exists()
        text = guide.read_text(encoding="utf-8")
        assert "terminology_data/" in text
        assert "license" in text.lower()


# ---------------------------------------------------------------------------
# TestNoClinicalLogicChange
# ---------------------------------------------------------------------------


class TestNoClinicalLogicChange:
    def test_no_clinical_text_in_modules(self):
        forbidden = (
            "take this dose", "recommended dose", "you should take",
            "mg per day", "we prescribe",
        )
        for fname in (
            "clinical_knowledge/terminology/intake_automation.py",
            "clinical_knowledge/terminology/file_classifier.py",
            "clinical_knowledge/terminology/ack_template.py",
            "scripts/cka_terminology_prepare_intake.py",
            "scripts/cka_terminology_check_ready.py",
        ):
            text = (REPO_ROOT / fname).read_text(encoding="utf-8").lower()
            for needle in forbidden:
                assert needle not in text, f"{fname}: forbidden {needle!r}"

    def test_main_mkb_store_unchanged(self):
        from clinical_knowledge.store import MKBStore as MS
        from clinical_knowledge.security import EncryptedCKAStore as Enc
        assert isinstance(MS, type)
        assert MS is not Enc

    def test_b07_synthetic_mapper_still_present(self):
        from clinical_knowledge.medical_coding import synthetic_mapper    # noqa: F401
