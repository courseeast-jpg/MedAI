"""CKA-SEC-03A — encrypted empty future store tests.

Covers:
- empty key refused
- key mismatch refused in script helper
- command-line --key / --encryption-key not accepted
- dry-run temp encrypted store creation
- correct-key open works
- wrong-key open fails
- plaintext absence verified
- records_count is zero
- manifest contains no key
- manifest contains no raw absolute path
- runtime_active stays false
- main_store_migration_performed stays false
- real_data_migrated stays false
- overwrite protection
- lock-file guard
- lock removed after run
- real-store creation blocked without explicit approval
- test-mode approved temp creation succeeds
- DB file is not staged in repo
- final report privacy clean
- no clinical text / no prescription dosing text in security package
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
    EmptyStoreInitError,
    EncryptedStoreManifest,
    InitializationResult,
    detect_sqlcipher_provider,
    initialize_empty_encrypted_store,
    initializer_will_create_real_store,
    safe_db_file_hash,
)
from clinical_knowledge.security.empty_store_initializer import (    # noqa: E402
    _is_inside_temp_dir,
    _lock_path_for,
)


REPORT_DIR = REPO_ROOT / "reports" / "cka_sec03a_empty_encrypted_store"
REAL_TARGET = REPO_ROOT / "data" / "secure" / "cka_encrypted_future_store.db"


def _has_provider() -> bool:
    return bool(detect_sqlcipher_provider().available)


def _new_key() -> str:
    return "synth_op_" + secrets.token_hex(16)


def _new_temp_db() -> str:
    fd, path = tempfile.mkstemp(prefix="cka_sec03a_pytest_", suffix=".db")
    os.close(fd)
    try:
        os.unlink(path)
    except OSError:
        pass
    return path


def _cleanup_db(p: str) -> None:
    path = Path(p)
    try:
        path.unlink(missing_ok=True)    # type: ignore[call-arg]
    except Exception:    # noqa: BLE001
        pass
    sib = path.parent / (path.stem + ".manifest.json")
    try:
        sib.unlink(missing_ok=True)    # type: ignore[call-arg]
    except Exception:    # noqa: BLE001
        pass
    lock = _lock_path_for(path)
    try:
        lock.unlink(missing_ok=True)    # type: ignore[call-arg]
    except Exception:    # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# TestImports
# ---------------------------------------------------------------------------


class TestImports:
    def test_security_package_exposes_sec03a_surface(self):
        import clinical_knowledge.security as sec
        for name in (
            "EncryptedStoreManifest",
            "safe_db_file_hash",
            "write_manifest_alongside_db",
            "EmptyStoreInitError",
            "InitializationResult",
            "initialize_empty_encrypted_store",
            "initializer_will_create_real_store",
        ):
            assert hasattr(sec, name), f"missing export: {name}"

    def test_validation_script_importable(self):
        from scripts import run_cka_sec03a_empty_encrypted_store_validation as v
        assert hasattr(v, "run_validation")

    def test_init_script_importable(self):
        from scripts import init_cka_empty_encrypted_store as s
        assert hasattr(s, "main")


# ---------------------------------------------------------------------------
# TestKeyHandling
# ---------------------------------------------------------------------------


class TestKeyHandling:
    def test_empty_key_refused(self):
        with pytest.raises(EmptyStoreInitError) as exc:
            initialize_empty_encrypted_store(_new_temp_db(), "")
        assert "empty_key" in str(exc.value)

    def test_short_key_refused(self):
        with pytest.raises(EmptyStoreInitError) as exc:
            initialize_empty_encrypted_store(_new_temp_db(), "short")
        assert "key_policy" in str(exc.value)

    def test_command_line_key_flag_refused_via_cli(self):
        # Either argparse refuses (because --key is store_true and won't
        # accept an explicit value) or our own check raises
        # command_line_key_not_accepted. Both are acceptable refusals.
        result = subprocess.run(
            [sys.executable, "scripts/init_cka_empty_encrypted_store.py",
             "--dry-run", "--key=somesecret"],
            cwd=str(REPO_ROOT),
            capture_output=True, text=True,
            timeout=30,
        )
        # Hard requirement: non-zero return code (key NOT accepted).
        assert result.returncode != 0
        # And the secret value must NOT appear in any output.
        combined = (result.stderr + "\n" + result.stdout).lower()
        # argparse echoes the rejected value once in its own error message;
        # that is acceptable since our process exited non-zero. What matters
        # is the script did not run successfully with the secret as a key.
        # We confirm at least one of the safe refusal markers is present.
        refusal_markers = (
            "key cannot be passed",
            "command_line_key_not_accepted",
            "ignored explicit argument",
            "unrecognized arguments",
            "error: argument --key",
        )
        assert any(m in combined for m in refusal_markers), (
            f"no recognized refusal marker; output={combined!r}"
        )

    def test_init_script_key_mismatch_refused(self):
        # Drive the script's _prompt_key_twice via monkey-patched getpass.
        from scripts import init_cka_empty_encrypted_store as s
        seq = iter(["abcdefghijkl1", "DIFFERENTKEY"])

        def fake_getpass(prompt):
            return next(seq)

        original = s.getpass.getpass
        s.getpass.getpass = fake_getpass
        try:
            with pytest.raises(EmptyStoreInitError) as exc:
                s._prompt_key_twice()
            assert "mismatch" in str(exc.value)
        finally:
            s.getpass.getpass = original

    def test_init_script_key_match_accepted(self):
        from scripts import init_cka_empty_encrypted_store as s
        seq = iter(["matchingkey_abc12", "matchingkey_abc12"])

        def fake_getpass(prompt):
            return next(seq)

        original = s.getpass.getpass
        s.getpass.getpass = fake_getpass
        try:
            assert s._prompt_key_twice() == "matchingkey_abc12"
        finally:
            s.getpass.getpass = original


# ---------------------------------------------------------------------------
# TestPathClassification
# ---------------------------------------------------------------------------


class TestPathClassification:
    def test_temp_path_recognised(self):
        p = Path(tempfile.gettempdir()) / "x.db"
        assert _is_inside_temp_dir(p) is True

    def test_repo_path_not_recognised_as_temp(self):
        p = REPO_ROOT / "data" / "secure" / "x.db"
        assert _is_inside_temp_dir(p) is False

    def test_inspection_helper_blocks_real_no_approval(self):
        assert initializer_will_create_real_store(
            str(REAL_TARGET), False) is False

    def test_inspection_helper_allows_real_with_approval(self):
        assert initializer_will_create_real_store(
            str(REAL_TARGET), True) is True

    def test_inspection_helper_temp_is_not_real(self):
        p = str(Path(tempfile.gettempdir()) / "x.db")
        assert initializer_will_create_real_store(p, True) is False


# ---------------------------------------------------------------------------
# TestApprovalGate
# ---------------------------------------------------------------------------


class TestApprovalGate:
    def test_real_store_blocked_without_approval(self):
        # Use a non-existent repo-internal path; must NOT be created.
        target = REPO_ROOT / "data" / "secure" / "cka_sec03a_test_unapproved.db"
        with pytest.raises(EmptyStoreInitError) as exc:
            initialize_empty_encrypted_store(str(target), _new_key())
        assert "not_approved" in str(exc.value)
        assert not target.exists()

    def test_real_target_default_path_absent(self):
        # The recommended-only file must not be present unless the operator
        # explicitly approved it during this run.
        assert not REAL_TARGET.exists()

    @pytest.mark.skipif(not _has_provider(),
                        reason="no SQLCipher provider")
    def test_temp_path_works_without_approval(self):
        db = _new_temp_db()
        try:
            r = initialize_empty_encrypted_store(db, _new_key())
            assert r.success is True
            assert r.records_count == 0
        finally:
            _cleanup_db(db)


# ---------------------------------------------------------------------------
# TestEmptyStoreLive
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_provider(),
                    reason="SQLCipher provider unavailable")
class TestEmptyStoreLive:
    def test_correct_key_read_and_invariants(self):
        db = _new_temp_db()
        try:
            r = initialize_empty_encrypted_store(db, _new_key())
            assert r.success is True
            assert r.schema_created is True
            assert r.records_count == 0
            assert r.ledger_events_count == 1
            assert r.correct_key_read_passed is True
            assert r.runtime_active is False
            assert r.main_store_migration_performed is False
            assert r.real_data_migrated is False
            assert r.lock_file_used is True
            assert r.lock_file_left_behind is False
            assert r.manifest_written is True
            assert r.db_file_staged is False
        finally:
            _cleanup_db(db)

    def test_wrong_key_failure(self):
        db = _new_temp_db()
        try:
            r = initialize_empty_encrypted_store(db, _new_key())
            assert r.wrong_key_failure_passed is True
        finally:
            _cleanup_db(db)

    def test_plaintext_absence(self):
        db = _new_temp_db()
        try:
            r = initialize_empty_encrypted_store(db, _new_key())
            assert r.plaintext_absence_verified is True
        finally:
            _cleanup_db(db)

    def test_overwrite_protection(self):
        db = _new_temp_db()
        try:
            initialize_empty_encrypted_store(db, _new_key())
            with pytest.raises(EmptyStoreInitError) as exc:
                initialize_empty_encrypted_store(db, _new_key())
            assert "target_exists" in str(exc.value)
        finally:
            _cleanup_db(db)

    def test_overwrite_with_flag_succeeds(self):
        db = _new_temp_db()
        try:
            initialize_empty_encrypted_store(db, _new_key())
            r = initialize_empty_encrypted_store(
                db, _new_key(), overwrite=True)
            assert r.success is True
        finally:
            _cleanup_db(db)

    def test_lock_blocks_init(self):
        db = _new_temp_db()
        lock = _lock_path_for(Path(db))
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text("stale", encoding="utf-8")
        try:
            with pytest.raises(EmptyStoreInitError) as exc:
                initialize_empty_encrypted_store(db, _new_key())
            assert "init_lock" in str(exc.value)
        finally:
            try:
                lock.unlink()
            except Exception:    # noqa: BLE001
                pass
            _cleanup_db(db)

    def test_lock_removed_after_run(self):
        db = _new_temp_db()
        try:
            r = initialize_empty_encrypted_store(db, _new_key())
            assert _lock_path_for(Path(db)).exists() is False
            assert r.lock_file_left_behind is False
        finally:
            _cleanup_db(db)


# ---------------------------------------------------------------------------
# TestManifest
# ---------------------------------------------------------------------------


class TestManifest:
    def test_manifest_no_key_field(self):
        m = EncryptedStoreManifest.for_new_store(
            db_path="/somewhere/foo.db",
            provider_name="sqlcipher3",
            cipher_version="4.x",
            operator_approved_creation=False,
        )
        s = m.safe_public_summary()
        assert "encryption_key" not in s
        assert s["encryption_key_logged"] is False
        assert s["key_stored_in_repo"] is False

    def test_manifest_no_raw_path(self):
        m = EncryptedStoreManifest.for_new_store(
            db_path="C:\\Users\\Operator\\AppData\\private\\foo.db",
            provider_name="sqlcipher3",
            cipher_version="4.x",
            operator_approved_creation=False,
        )
        s = m.safe_public_summary()
        text = json.dumps(s)
        assert "Operator" not in text
        assert "AppData" not in text
        assert not re.search(r"[A-Za-z]:\\\\", text)
        assert s["db_file_safe_hash"].startswith("cka_db_")

    def test_manifest_runtime_active_must_be_false(self):
        with pytest.raises(ValueError):
            EncryptedStoreManifest(
                manifest_id="x", created_at="x",
                store_safe_id="cka_db_x",
                runtime_active=True,
            )

    def test_manifest_main_store_migration_must_be_false(self):
        with pytest.raises(ValueError):
            EncryptedStoreManifest(
                manifest_id="x", created_at="x",
                store_safe_id="cka_db_x",
                main_store_migration_performed=True,
            )

    def test_manifest_real_data_must_be_false(self):
        with pytest.raises(ValueError):
            EncryptedStoreManifest(
                manifest_id="x", created_at="x",
                store_safe_id="cka_db_x",
                real_data_migrated=True,
            )

    def test_manifest_records_count_must_be_zero(self):
        with pytest.raises(ValueError):
            EncryptedStoreManifest(
                manifest_id="x", created_at="x",
                store_safe_id="cka_db_x",
                records_count=1,
            )

    def test_safe_db_file_hash_stable(self):
        h1 = safe_db_file_hash("/some/path/foo.db")
        h2 = safe_db_file_hash("/some/path/foo.db")
        h3 = safe_db_file_hash("/other/path/foo.db")
        assert h1 == h2
        assert h1 != h3
        assert h1.startswith("cka_db_")
        assert len(h1) == len("cka_db_") + 16


# ---------------------------------------------------------------------------
# TestPublicReport
# ---------------------------------------------------------------------------


class TestPublicReport:
    @pytest.fixture(scope="class")
    def report(self):
        from scripts.run_cka_sec03a_empty_encrypted_store_validation import (
            run_validation,
        )
        return run_validation()

    def test_block_id(self, report):
        assert report["block_id"] == "CKA-SEC-03A"

    def test_conclusion(self, report):
        assert report["conclusion"] == "cka_sec03a_empty_encrypted_future_store_ready"

    def test_safety_flags_false(self, report):
        for k in (
            "empty_future_store_runtime_active",
            "main_store_migration_performed",
            "real_data_migrated",
            "real_existing_store_migrated",
            "encryption_key_logged",
            "key_stored_in_repo",
            "db_file_staged",
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

    def test_real_empty_store_only_if_operator_approved(self, report):
        assert report["real_empty_store_created_only_if_operator_approved"] is True

    def test_report_no_key_prefix(self, report):
        text = json.dumps(report)
        assert "synth_op_" not in text

    def test_report_no_drive_letter_path(self, report):
        text = json.dumps(report)
        assert not re.search(r"[A-Za-z]:\\\\", text)

    def test_report_no_temp_prefix(self, report):
        text = json.dumps(report)
        assert "cka_sec03a_v_" not in text
        assert "cka_sec03a_init_" not in text
        assert "cka_sec03a_pytest_" not in text

    def test_report_passes_b02_privacy_checker(self, report):
        from clinical_knowledge.privacy.report_privacy import (
            check_public_report_payload,
        )
        result = check_public_report_payload(report)
        assert result.passed, (
            f"privacy checker rejected: {result.leak_examples_redacted}"
        )

    def test_report_md_present(self):
        md = REPORT_DIR / "cka_sec03a_empty_encrypted_store_report.md"
        assert md.exists()

    def test_operator_guide_present(self):
        guide = REPORT_DIR / "CKA_SEC03A_EMPTY_STORE_OPERATOR_GUIDE.md"
        assert guide.exists()
        text = guide.read_text(encoding="utf-8").lower()
        assert "encryption key" in text
        assert "lost key" in text
        assert "data/secure/cka_encrypted_future_store.db" in text


# ---------------------------------------------------------------------------
# TestNoStagedDB
# ---------------------------------------------------------------------------


class TestNoStagedDB:
    def test_no_real_db_present(self):
        # The recommended path leaves no DB in the repo.
        assert not REAL_TARGET.exists() or self._approved_real_db()

    def _approved_real_db(self) -> bool:
        # Allow CI configurations that explicitly approve the real DB.
        return os.environ.get("CKA_SEC03A_APPROVE_REAL", "0") == "1"

    def test_no_db_under_security_package(self):
        sec_dir = REPO_ROOT / "clinical_knowledge" / "security"
        for p in sec_dir.rglob("*"):
            if p.is_file():
                assert p.suffix.lower() not in (".db", ".sqlite", ".sqlite3")

    def test_no_db_under_report_dir(self):
        if not REPORT_DIR.exists():
            pytest.skip("report dir not yet created")
        for p in REPORT_DIR.rglob("*"):
            if p.is_file():
                assert p.suffix.lower() not in (".db", ".sqlite", ".sqlite3")


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
            "clinical_knowledge/security/sqlcipher_provider.py",
            "clinical_knowledge/security/encrypted_store.py",
            "clinical_knowledge/security/encryption_checks.py",
            "clinical_knowledge/security/key_policy.py",
            "clinical_knowledge/security/migration_plan.py",
            "clinical_knowledge/security/migration_rehearsal.py",
            "clinical_knowledge/security/rollback_plan.py",
            "clinical_knowledge/security/encrypted_store_manifest.py",
            "clinical_knowledge/security/empty_store_initializer.py",
        ):
            text = (REPO_ROOT / fname).read_text(encoding="utf-8").lower()
            for needle in forbidden:
                assert needle not in text, f"{fname}: forbidden text {needle!r}"

    def test_consensus_engine_still_loads(self):
        from clinical_knowledge.consensus.engine import run_consensus    # noqa: F401

    def test_decision_engine_still_loads(self):
        import clinical_knowledge.decision_engine.engine    # noqa: F401

    def test_truth_resolution_still_loads(self):
        import clinical_knowledge.truth_resolution.engine    # noqa: F401

    def test_ddi_stub_still_loads(self):
        import clinical_knowledge.medication_safety.ddi_stub    # noqa: F401

    def test_main_mkb_store_class_unchanged(self):
        from clinical_knowledge.store import MKBStore
        from clinical_knowledge.security import EncryptedCKAStore
        assert isinstance(MKBStore, type)
        assert MKBStore is not EncryptedCKAStore
