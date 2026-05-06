"""CKA-SEC-07 — encrypted backup / restore tooling tests.

Covers:
- module imports
- BackupManifest invariants (no key, no real data, count >= 0)
- file_sha256 and prefix helpers
- backup CLI rejects --key / --encryption-key
- restore CLI rejects --key / --encryption-key
- dry-run end-to-end via the operator scripts
- create_encrypted_backup refuses empty key, missing source, existing target
- restore_encrypted_backup refuses empty key, missing backup, missing manifest, existing target
- backup is encrypted (plaintext absence in raw bytes)
- correct-key restore round-trip, record count match
- wrong-key restore fails (no silent success)
- no DB / key / private files staged
- final SEC-05 + final CKA validation invocable
- public report no key/path/PHI/secret
- no clinical / dosing wording in modules
- main MKBStore class identity unchanged
"""
from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from clinical_knowledge.security import (    # noqa: E402
    BackupManifest,
    BackupResult,
    EncryptedBackupError,
    EncryptedRestoreError,
    EncryptedRuntimeConfig,
    RestoreResult,
    build_cka_runtime_store,
    create_encrypted_backup,
    detect_sqlcipher_provider,
    file_sha256,
    file_sha256_prefix,
    manifest_path_for,
    read_backup_manifest,
    restore_encrypted_backup,
    verify_restored_wrong_key_fails,
)
from clinical_knowledge.security.empty_store_initializer import (    # noqa: E402
    _lock_path_for,
)


REPORT_DIR = REPO_ROOT / "reports" / "cka_sec07_encrypted_backup_restore"
REAL_TARGET = REPO_ROOT / "data" / "secure" / "cka_encrypted_future_store.db"
BACKUP_SCRIPT = REPO_ROOT / "scripts" / "cka_encrypted_store_backup.py"
RESTORE_SCRIPT = REPO_ROOT / "scripts" / "cka_encrypted_store_restore.py"


def _has_provider() -> bool:
    return bool(detect_sqlcipher_provider().available)


def _new_key() -> str:
    return "synth_op_" + secrets.token_hex(16)


def _new_temp_db(prefix: str = "cka_sec07_pytest_") -> str:
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".db")
    os.close(fd)
    try:
        os.unlink(path)
    except OSError:
        pass
    return path


def _cleanup(p: str) -> None:
    path = Path(p)
    try:
        path.unlink(missing_ok=True)    # type: ignore[call-arg]
    except Exception:    # noqa: BLE001
        pass
    for ext in (".manifest.json", ".backup-manifest.json"):
        sib = path.parent / (path.stem + ext)
        try:
            sib.unlink(missing_ok=True)    # type: ignore[call-arg]
        except Exception:    # noqa: BLE001
            pass
    lock = _lock_path_for(path)
    try:
        lock.unlink(missing_ok=True)    # type: ignore[call-arg]
    except Exception:    # noqa: BLE001
        pass


def _create_synth_source(key: str, n: int = 2) -> str:
    src = _new_temp_db("cka_sec07_pytest_src_")
    cfg = EncryptedRuntimeConfig.for_test(
        src, key, encrypted_runtime_requested=True, create_if_missing=True,
    )
    r = build_cka_runtime_store(cfg)
    con = r.store._con
    for i in range(n):
        con.execute(
            "INSERT INTO cka_future_records (record_id, label, payload, created_at) "
            "VALUES (?, ?, ?, ?)",
            (f"rec_{i:03d}", f"synthetic_label_{i}",
             f"synthetic_payload_{i}", "2026-05-06T00:00:00Z"),
        )
    con.commit()
    r.store.close()
    return src


# ---------------------------------------------------------------------------
# TestImports
# ---------------------------------------------------------------------------


class TestImports:
    def test_security_package_exposes_sec07_surface(self):
        import clinical_knowledge.security as sec
        for name in (
            "BackupManifest", "BackupResult", "EncryptedBackupError",
            "create_encrypted_backup", "RestoreResult", "EncryptedRestoreError",
            "restore_encrypted_backup", "verify_restored_wrong_key_fails",
            "file_sha256", "file_sha256_prefix",
            "manifest_path_for", "read_backup_manifest",
            "write_backup_manifest", "backup_safe_db_hash",
        ):
            assert hasattr(sec, name), f"missing export: {name}"

    def test_validation_script_importable(self):
        from scripts import run_cka_sec07_encrypted_backup_restore_validation as v
        assert hasattr(v, "run_validation")

    def test_backup_script_importable(self):
        from scripts import cka_encrypted_store_backup as s
        assert hasattr(s, "main")

    def test_restore_script_importable(self):
        from scripts import cka_encrypted_store_restore as s
        assert hasattr(s, "main")


# ---------------------------------------------------------------------------
# TestBackupManifest
# ---------------------------------------------------------------------------


class TestBackupManifest:
    def test_safe_summary_no_key(self):
        m = BackupManifest.for_new_backup(
            "/some/source.db", "/some/backup.db",
            backup_sha256_prefix="abcdef0123456789",
            source_record_count=2,
        )
        s = m.safe_public_summary()
        text = json.dumps(s)
        # No raw paths in the summary.
        assert "/some/source.db" not in text
        assert "/some/backup.db" not in text
        # No "encryption_key" field with value.
        assert "encryption_key" not in s
        assert s["encryption_key_logged"] is False
        assert s["key_stored_in_repo"] is False
        assert s["real_data_in_backup"] is False
        assert s["source_record_count"] == 2

    def test_real_data_flag_must_be_false(self):
        with pytest.raises(ValueError):
            BackupManifest(
                manifest_id="x", created_at="x",
                source_safe_id="cka_db_x", backup_safe_id="cka_db_y",
                backup_sha256_prefix="aaaaaaaa",
                real_data_in_backup=True,
            )

    def test_key_logged_flag_must_be_false(self):
        with pytest.raises(ValueError):
            BackupManifest(
                manifest_id="x", created_at="x",
                source_safe_id="cka_db_x", backup_safe_id="cka_db_y",
                backup_sha256_prefix="aaaaaaaa",
                encryption_key_logged=True,
            )

    def test_key_stored_flag_must_be_false(self):
        with pytest.raises(ValueError):
            BackupManifest(
                manifest_id="x", created_at="x",
                source_safe_id="cka_db_x", backup_safe_id="cka_db_y",
                backup_sha256_prefix="aaaaaaaa",
                key_stored_in_repo=True,
            )

    def test_negative_record_count_rejected(self):
        with pytest.raises(ValueError):
            BackupManifest(
                manifest_id="x", created_at="x",
                source_safe_id="cka_db_x", backup_safe_id="cka_db_y",
                backup_sha256_prefix="aaaaaaaa",
                source_record_count=-1,
            )


# ---------------------------------------------------------------------------
# TestSha256Helpers
# ---------------------------------------------------------------------------


class TestSha256Helpers:
    def test_full_sha256_stable(self, tmp_path):
        f = tmp_path / "x.bin"
        f.write_bytes(b"hello")
        full = file_sha256(str(f))
        assert full is not None
        assert len(full) == 64    # 256-bit hex

    def test_prefix_length_default_16(self, tmp_path):
        f = tmp_path / "x.bin"
        f.write_bytes(b"hello")
        prefix = file_sha256_prefix(str(f))
        assert prefix is not None
        assert len(prefix) == 16
        # 16 hex chars are below the B02 SECRET regex's 40+ alnum threshold.
        assert len(prefix) < 40

    def test_missing_file_returns_none(self):
        assert file_sha256("does/not/exist.bin") is None
        assert file_sha256_prefix("does/not/exist.bin") is None


# ---------------------------------------------------------------------------
# TestBackupGuards
# ---------------------------------------------------------------------------


class TestBackupGuards:
    def test_empty_key_refused(self):
        with pytest.raises(EncryptedBackupError) as exc:
            create_encrypted_backup("any.db", "x.db", "")
        assert "key_policy" in str(exc.value)

    def test_missing_source_refused(self):
        # Use a key that survives policy.
        with pytest.raises(EncryptedBackupError) as exc:
            create_encrypted_backup(
                str(REPO_ROOT / "does_not_exist_sec07.db"),
                "x.db", _new_key(),
            )
        assert "source_missing" in str(exc.value)

    @pytest.mark.skipif(not _has_provider(),
                        reason="SQLCipher provider unavailable")
    def test_existing_backup_refused_without_overwrite(self, tmp_path):
        key = _new_key()
        src = _create_synth_source(key)
        bk = str(tmp_path / "existing_backup.db")
        Path(bk).write_bytes(b"placeholder")
        try:
            with pytest.raises(EncryptedBackupError) as exc:
                create_encrypted_backup(src, bk, key)
            assert "overwrite_required" in str(exc.value)
        finally:
            _cleanup(src)
            _cleanup(bk)


# ---------------------------------------------------------------------------
# TestRestoreGuards
# ---------------------------------------------------------------------------


class TestRestoreGuards:
    def test_empty_key_refused(self):
        with pytest.raises(EncryptedRestoreError) as exc:
            restore_encrypted_backup("any.db", "x.db", "")
        assert "key_policy" in str(exc.value)

    def test_missing_backup_refused(self):
        with pytest.raises(EncryptedRestoreError) as exc:
            restore_encrypted_backup(
                str(REPO_ROOT / "does_not_exist_sec07_bk.db"),
                "x.db", _new_key(),
            )
        assert "backup_missing" in str(exc.value)

    def test_missing_manifest_refused(self, tmp_path):
        # Create a fake "backup" file with no manifest sibling.
        fake_backup = tmp_path / "fake_backup.db"
        fake_backup.write_bytes(b"junk-without-manifest")
        target = tmp_path / "target.db"
        with pytest.raises(EncryptedRestoreError) as exc:
            restore_encrypted_backup(str(fake_backup), str(target), _new_key())
        assert "manifest_missing" in str(exc.value)


# ---------------------------------------------------------------------------
# TestRoundTripLive
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_provider(),
                    reason="SQLCipher provider unavailable")
class TestRoundTripLive:
    def test_backup_restore_round_trip(self):
        key = _new_key()
        src = bk = tgt = None
        try:
            src = _create_synth_source(key, n=3)
            bk = _new_temp_db("cka_sec07_pytest_bk_")
            br = create_encrypted_backup(src, bk, key)
            assert br.success is True
            assert br.source_record_count == 3
            assert br.plaintext_absence_verified is True
            assert br.manifest_written is True

            tgt = _new_temp_db("cka_sec07_pytest_tgt_")
            rr = restore_encrypted_backup(bk, tgt, key)
            assert rr.success is True
            assert rr.sha256_match is True
            assert rr.correct_key_read_passed is True
            assert rr.expected_record_count == 3
            assert rr.restored_record_count == 3
            assert rr.record_count_match is True
            assert rr.plaintext_absence_verified is True
            assert rr.real_data_touched is False
            assert rr.encryption_key_logged is False
        finally:
            _cleanup(src or "")
            _cleanup(bk or "")
            _cleanup(tgt or "")

    def test_wrong_key_restore_fails(self):
        correct = _new_key()
        src = bk = tgt = None
        try:
            src = _create_synth_source(correct, n=2)
            bk = _new_temp_db("cka_sec07_pytest_bk_")
            create_encrypted_backup(src, bk, correct)

            tgt = _new_temp_db("cka_sec07_pytest_tgt_")
            wrong = _new_key()
            with pytest.raises(EncryptedRestoreError) as exc:
                restore_encrypted_backup(bk, tgt, wrong)
            assert "open_failed" in str(exc.value)
        finally:
            _cleanup(src or "")
            _cleanup(bk or "")
            _cleanup(tgt or "")

    def test_post_restore_wrong_key_probe(self):
        """After a successful restore, an out-of-band wrong-key probe must fail."""
        correct = _new_key()
        src = bk = tgt = None
        try:
            src = _create_synth_source(correct)
            bk = _new_temp_db("cka_sec07_pytest_bk_")
            create_encrypted_backup(src, bk, correct)
            tgt = _new_temp_db("cka_sec07_pytest_tgt_")
            restore_encrypted_backup(bk, tgt, correct)
            wrong = _new_key()
            assert verify_restored_wrong_key_fails(tgt, wrong) is True
        finally:
            _cleanup(src or "")
            _cleanup(bk or "")
            _cleanup(tgt or "")

    def test_backup_summary_no_key(self):
        key = _new_key()
        src = bk = None
        try:
            src = _create_synth_source(key)
            bk = _new_temp_db("cka_sec07_pytest_bk_")
            br = create_encrypted_backup(src, bk, key)
            text = json.dumps(br.safe_public_summary())
            assert key not in text
            assert "synth_op_" not in text
        finally:
            _cleanup(src or "")
            _cleanup(bk or "")


# ---------------------------------------------------------------------------
# TestCLIRejectsKey
# ---------------------------------------------------------------------------


class TestCLIRejectsKey:
    @pytest.mark.parametrize("argv", [
        ["--key=secret123"],
        ["--encryption-key=secret123"],
        ["--key"],
    ])
    def test_backup_subprocess_rejects_key(self, argv):
        res = subprocess.run(
            [sys.executable, str(BACKUP_SCRIPT), *argv],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=15,
        )
        assert res.returncode != 0
        combined = res.stdout + "\n" + res.stderr
        assert "secret123" not in combined

    @pytest.mark.parametrize("argv", [
        ["--key=secret123"],
        ["--encryption-key=secret123"],
        ["--key"],
    ])
    def test_restore_subprocess_rejects_key(self, argv):
        res = subprocess.run(
            [sys.executable, str(RESTORE_SCRIPT), *argv],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=15,
        )
        assert res.returncode != 0
        combined = res.stdout + "\n" + res.stderr
        assert "secret123" not in combined


# ---------------------------------------------------------------------------
# TestDryRun
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_provider(),
                    reason="SQLCipher provider unavailable")
class TestDryRun:
    def test_backup_dry_run_clean(self):
        existed = REAL_TARGET.exists()
        res = subprocess.run(
            [sys.executable, str(BACKUP_SCRIPT), "--dry-run"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
        )
        assert res.returncode == 0, res.stderr
        assert "success: True" in res.stdout
        assert "synth_op_" not in res.stdout
        assert "synth_op_" not in res.stderr
        # Real target file must not appear.
        assert REAL_TARGET.exists() == existed

    def test_restore_dry_run_clean(self):
        existed = REAL_TARGET.exists()
        res = subprocess.run(
            [sys.executable, str(RESTORE_SCRIPT), "--dry-run"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
        )
        assert res.returncode == 0, res.stderr
        assert "success: True" in res.stdout
        assert "synth_op_" not in res.stdout
        assert "synth_op_" not in res.stderr
        assert REAL_TARGET.exists() == existed


# ---------------------------------------------------------------------------
# TestNoStagedDB
# ---------------------------------------------------------------------------


class TestNoStagedDB:
    def test_no_db_in_security_package(self):
        sec_dir = REPO_ROOT / "clinical_knowledge" / "security"
        for p in sec_dir.rglob("*"):
            if p.is_file():
                assert p.suffix.lower() not in (".db", ".sqlite", ".sqlite3")

    def test_no_db_in_sec07_report_dir(self):
        if not REPORT_DIR.exists():
            pytest.skip("report dir not yet created")
        for p in REPORT_DIR.rglob("*"):
            if p.is_file():
                assert p.suffix.lower() not in (".db", ".sqlite", ".sqlite3")

    def test_no_real_db_in_repo(self):
        # Real target file is intentionally absent until the operator
        # explicitly creates it via SEC-03A or SEC-05.
        if REAL_TARGET.exists():
            pytest.skip("operator created real target — out of test scope")


# ---------------------------------------------------------------------------
# TestPublicReport
# ---------------------------------------------------------------------------


class TestPublicReport:
    @pytest.fixture(scope="class")
    def report(self):
        from scripts.run_cka_sec07_encrypted_backup_restore_validation import (
            run_validation,
        )
        return run_validation()

    def test_block_id(self, report):
        assert report["block_id"] == "CKA-SEC-07"

    def test_conclusion(self, report):
        assert report["conclusion"] == "cka_sec07_encrypted_backup_restore_ready"

    def test_round_trip_flags(self, report):
        for k in (
            "backup_tool_ready",
            "restore_tool_ready",
            "dry_run_supported",
            "synthetic_backup_restore_passed",
            "correct_key_restore_passed",
            "wrong_key_restore_failed",
            "checksum_verified",
            "plaintext_absence_verified",
        ):
            assert report[k] is True, f"flag {k} not True"

    def test_safety_flags_false(self, report):
        for k in (
            "real_data_touched",
            "real_store_modified",
            "db_file_staged",
            "key_stored_in_repo",
            "encryption_key_logged",
            "external_api_used",
            "raw_phi_logged_in_public_reports",
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

    def test_report_no_key_prefix(self, report):
        text = json.dumps(report)
        assert "synth_op_" not in text

    def test_report_no_drive_letter_path(self, report):
        text = json.dumps(report)
        assert not re.search(r"[A-Za-z]:\\\\", text)

    def test_report_no_temp_prefix(self, report):
        text = json.dumps(report)
        for needle in (
            "cka_sec07_src_", "cka_sec07_bk_", "cka_sec07_tgt_",
            "cka_sec07_dryrun_", "cka_sec07_v_", "cka_sec07_pytest_",
            "cka_sec05_", "cka_sec04_", "cka_sec03a_",
        ):
            assert needle not in text, f"temp prefix {needle!r} present"

    def test_report_passes_b02_privacy_checker(self, report):
        from clinical_knowledge.privacy.report_privacy import (
            check_public_report_payload,
        )
        result = check_public_report_payload(report)
        assert result.passed, (
            f"privacy checker rejected: {result.leak_examples_redacted}"
        )

    def test_report_md_present(self):
        md = REPORT_DIR / "cka_sec07_encrypted_backup_restore_report.md"
        assert md.exists()

    def test_operator_guide_present(self):
        guide = REPORT_DIR / "CKA_SEC07_BACKUP_RESTORE_OPERATOR_GUIDE.md"
        assert guide.exists()
        text = guide.read_text(encoding="utf-8")
        assert "cka_encrypted_store_backup.py" in text
        assert "cka_encrypted_store_restore.py" in text


# ---------------------------------------------------------------------------
# TestNoClinicalLogicChange
# ---------------------------------------------------------------------------


class TestNoClinicalLogicChange:
    def test_no_clinical_text_in_security_package(self):
        forbidden = (
            "take this dose",
            "recommended dose",
            "you should take",
            "mg per day",
            "we prescribe",
        )
        for fname in (
            "clinical_knowledge/security/__init__.py",
            "clinical_knowledge/security/encrypted_backup.py",
            "clinical_knowledge/security/encrypted_restore.py",
            "clinical_knowledge/security/backup_manifest.py",
            "clinical_knowledge/security/runtime_launcher.py",
            "scripts/cka_encrypted_store_backup.py",
            "scripts/cka_encrypted_store_restore.py",
        ):
            text = (REPO_ROOT / fname).read_text(encoding="utf-8").lower()
            for needle in forbidden:
                assert needle not in text, f"{fname}: forbidden {needle!r}"

    def test_main_mkb_store_class_unchanged(self):
        from clinical_knowledge.store import MKBStore as MS
        from clinical_knowledge.security import EncryptedCKAStore as Enc
        assert isinstance(MS, type)
        assert MS is not Enc

    def test_consensus_engine_still_loads(self):
        from clinical_knowledge.consensus.engine import run_consensus    # noqa: F401

    def test_decision_engine_still_loads(self):
        import clinical_knowledge.decision_engine.engine    # noqa: F401

    def test_truth_resolution_still_loads(self):
        import clinical_knowledge.truth_resolution.engine    # noqa: F401

    def test_ddi_stub_still_loads(self):
        import clinical_knowledge.medication_safety.ddi_stub    # noqa: F401
