"""CKA-SEC-06 — operator key rotation tests.

Covers:
- model imports + safe-public-summary contains no key/path
- KeyRotationPlan defaults safe; backup_required cannot be False
- KeyRotationResult fields default to non-touched / non-staged
- CLI subprocess rejects --old-key / --new-key / --key / --encryption-key
- prompt_old_key / prompt_new_key_twice helpers refuse mismatch + empty
- rotate_sqlcipher_key refuses empty key, same old/new, missing source
- non-temp path refused without approve_real_rotation
- synthetic rehearsal end-to-end (rotation, count preserved, old key rejected,
  rollback restore, plaintext absence, no temp files staged)
- public report no key/path/temp-prefix/PHI; B02 privacy checker passes
- no DB / key / private files staged in security package or report dir
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
    KeyRotationError,
    KeyRotationPlan,
    KeyRotationResult,
    detect_sqlcipher_provider,
    key_rotation_rollback_steps,
    rotate_sqlcipher_key,
    rotation_passed,
    run_synthetic_rotation_rehearsal,
)
from clinical_knowledge.security.empty_store_initializer import (    # noqa: E402
    _lock_path_for,
)


REPORT_DIR = REPO_ROOT / "reports" / "cka_sec06_key_rotation"
REAL_TARGET = REPO_ROOT / "data" / "secure" / "cka_encrypted_future_store.db"
ROTATE_SCRIPT = REPO_ROOT / "scripts" / "cka_encrypted_store_rotate_key.py"


def _has_provider() -> bool:
    return bool(detect_sqlcipher_provider().available)


def _new_key() -> str:
    return "synth_op_" + secrets.token_hex(16)


def _new_temp_db(prefix: str = "cka_sec06_pytest_") -> str:
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


# ---------------------------------------------------------------------------
# TestImports
# ---------------------------------------------------------------------------


class TestImports:
    def test_security_package_exposes_sec06_surface(self):
        import clinical_knowledge.security as sec
        for name in (
            "KeyRotationPlan", "KeyRotationResult",
            "key_rotation_rollback_steps", "rotation_passed",
            "KeyRotationError", "rotate_sqlcipher_key",
            "run_synthetic_rotation_rehearsal",
        ):
            assert hasattr(sec, name), f"missing export: {name}"

    def test_validation_script_importable(self):
        from scripts import run_cka_sec06_key_rotation_validation as v
        assert hasattr(v, "run_validation")

    def test_rotate_script_importable(self):
        from scripts import cka_encrypted_store_rotate_key as s
        assert hasattr(s, "main")
        assert hasattr(s, "prompt_old_key")
        assert hasattr(s, "prompt_new_key_twice")


# ---------------------------------------------------------------------------
# TestKeyRotationPlan
# ---------------------------------------------------------------------------


class TestKeyRotationPlan:
    def test_dry_run_defaults_safe(self):
        plan = KeyRotationPlan.for_dry_run("cka_db_x", test_mode=True)
        assert plan.dry_run is True
        assert plan.test_mode is True
        assert plan.real_rotation_approved is False
        assert plan.key_rotation_performed is False
        assert plan.real_store_touched is False
        assert plan.backup_required is True
        assert plan.backup_before_rotation_required is True
        assert plan.new_key_confirmation_required is True

    def test_backup_required_false_raises(self):
        with pytest.raises(ValueError):
            KeyRotationPlan(
                plan_id="x", created_at="x", source_store_safe_id="cka_db_x",
                backup_required=False,
            )

    def test_backup_before_rotation_false_raises(self):
        with pytest.raises(ValueError):
            KeyRotationPlan(
                plan_id="x", created_at="x", source_store_safe_id="cka_db_x",
                backup_before_rotation_required=False,
            )

    def test_new_key_confirmation_required_false_raises(self):
        with pytest.raises(ValueError):
            KeyRotationPlan(
                plan_id="x", created_at="x", source_store_safe_id="cka_db_x",
                new_key_confirmation_required=False,
            )

    def test_summary_no_key_or_path(self):
        plan = KeyRotationPlan.for_dry_run("cka_db_x")
        s = plan.safe_public_summary()
        text = json.dumps(s)
        assert "synth_op_" not in text
        assert "encryption_key" not in text


# ---------------------------------------------------------------------------
# TestKeyRotationResult
# ---------------------------------------------------------------------------


class TestKeyRotationResult:
    def test_default_safe(self):
        r = KeyRotationResult()
        s = r.safe_public_summary()
        # Defaults: nothing was performed and nothing was touched.
        assert s["rotation_performed"] is False
        assert s["real_store_touched"] is False
        assert s["db_file_staged"] is False
        assert s["key_logged"] is False

    def test_summary_no_keylike_strings(self):
        r = KeyRotationResult(
            source_safe_hash="cka_db_aaaaaaaaaaaaaaaa",
            backup_safe_hash="cka_db_bbbbbbbbbbbbbbbb",
            backup_sha256_prefix="abcdef0123456789",
        )
        s = r.safe_public_summary()
        text = json.dumps(s)
        # 16-char prefix is below the 40+-alnum SECRET regex threshold.
        assert len(s["backup_sha256_prefix"]) == 16
        assert "synth_op_" not in text


# ---------------------------------------------------------------------------
# TestRotationGuards
# ---------------------------------------------------------------------------


class TestRotationGuards:
    def test_empty_old_key_blocked(self):
        r = rotate_sqlcipher_key("any.db", "", _new_key())
        assert r.blocked_reason is not None
        assert "empty_key" in r.blocked_reason

    def test_empty_new_key_blocked(self):
        r = rotate_sqlcipher_key("any.db", _new_key(), "")
        assert r.blocked_reason is not None
        assert "empty_key" in r.blocked_reason

    def test_same_old_new_key_blocked(self):
        k = _new_key()
        r = rotate_sqlcipher_key("any.db", k, k)
        assert r.blocked_reason == "same_old_new_key_refused"

    def test_missing_source_blocked(self):
        r = rotate_sqlcipher_key(
            str(REPO_ROOT / "does_not_exist_sec06.db"),
            _new_key(), _new_key(),
        )
        assert r.blocked_reason == "source_missing"

    def test_real_path_blocked_without_approval(self, tmp_path):
        # We use a real file in the repo (under data/secure) that is NOT
        # in the system temp dir. Without approve_real_rotation, the
        # rotation must refuse and NOT modify the file.
        sentinel = REPO_ROOT / "data" / "secure" / "_sec06_pytest_sentinel.db"
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_bytes(b"placeholder-not-an-encrypted-db")
        try:
            before_size = sentinel.stat().st_size
            r = rotate_sqlcipher_key(
                str(sentinel),
                _new_key(), _new_key(),
                approve_real_rotation=False,
            )
            assert r.blocked_reason == "real_rotation_not_approved"
            assert sentinel.stat().st_size == before_size
        finally:
            sentinel.unlink(missing_ok=True)    # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# TestRehearsalLive
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_provider(),
                    reason="SQLCipher provider unavailable")
class TestRehearsalLive:
    def test_synthetic_rehearsal_round_trip(self):
        result, _src, _bkp = run_synthetic_rotation_rehearsal(record_count=3)
        assert rotation_passed(result) is True
        assert result.rotation_performed is True
        assert result.source_records_before == 3
        assert result.source_records_after == 3
        assert result.record_count_preserved is True
        assert result.old_key_open_before_passed is True
        assert result.new_key_open_after_passed is True
        assert result.old_key_rejected_after_rotation is True
        assert result.backup_created_before_rotation is True
        assert result.backup_checksum_verified is True
        assert result.rollback_restore_verified is True
        assert result.plaintext_absence_verified is True
        assert result.real_store_touched is False
        assert result.db_file_staged is False
        assert result.key_logged is False

    def test_rehearsal_summary_no_key(self):
        result, _src, _bkp = run_synthetic_rotation_rehearsal(record_count=2)
        s = result.safe_public_summary()
        text = json.dumps(s)
        assert "synth_op_" not in text
        # Backup checksum prefix must be short — 16 hex.
        assert len(s["backup_sha256_prefix"]) == 16


# ---------------------------------------------------------------------------
# TestPromptHelpers
# ---------------------------------------------------------------------------


class TestPromptHelpers:
    def test_prompt_old_key_match(self):
        from scripts import cka_encrypted_store_rotate_key as cli
        original = cli.getpass.getpass
        cli.getpass.getpass = lambda prompt: "matchingkey_001"
        try:
            assert cli.prompt_old_key() == "matchingkey_001"
        finally:
            cli.getpass.getpass = original

    def test_prompt_old_key_empty_refused(self):
        from scripts import cka_encrypted_store_rotate_key as cli
        original = cli.getpass.getpass
        cli.getpass.getpass = lambda prompt: ""
        try:
            with pytest.raises(cli.LauncherError) as exc:
                cli.prompt_old_key()
            assert "empty_old_key" in str(exc.value)
        finally:
            cli.getpass.getpass = original

    def test_prompt_new_key_twice_match(self):
        from scripts import cka_encrypted_store_rotate_key as cli
        seq = iter(["newkey_match_1", "newkey_match_1"])
        original = cli.getpass.getpass
        cli.getpass.getpass = lambda prompt: next(seq)    # type: ignore[assignment]
        try:
            assert cli.prompt_new_key_twice() == "newkey_match_1"
        finally:
            cli.getpass.getpass = original

    def test_prompt_new_key_twice_mismatch_refused(self):
        from scripts import cka_encrypted_store_rotate_key as cli
        seq = iter(["abcdefghijkl1", "DIFFERENTKEY9"])
        original = cli.getpass.getpass
        cli.getpass.getpass = lambda prompt: next(seq)    # type: ignore[assignment]
        try:
            with pytest.raises(cli.LauncherError) as exc:
                cli.prompt_new_key_twice()
            assert "mismatch" in str(exc.value)
        finally:
            cli.getpass.getpass = original

    def test_prompt_new_key_twice_empty_refused(self):
        from scripts import cka_encrypted_store_rotate_key as cli
        seq = iter([""])
        original = cli.getpass.getpass
        cli.getpass.getpass = lambda prompt: next(seq)    # type: ignore[assignment]
        try:
            with pytest.raises(cli.LauncherError) as exc:
                cli.prompt_new_key_twice()
            assert "empty_new_key" in str(exc.value)
        finally:
            cli.getpass.getpass = original


# ---------------------------------------------------------------------------
# TestCLIRejectsKey
# ---------------------------------------------------------------------------


class TestCLIRejectsKey:
    @pytest.mark.parametrize("argv", [
        ["--old-key=secret123"],
        ["--new-key=secret123"],
        ["--key=secret123"],
        ["--encryption-key=secret123"],
        ["--old-key"],
        ["--new-key"],
    ])
    def test_subprocess_rejects(self, argv):
        res = subprocess.run(
            [sys.executable, str(ROTATE_SCRIPT), *argv],
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
    def test_dry_run_clean(self):
        existed = REAL_TARGET.exists()
        res = subprocess.run(
            [sys.executable, str(ROTATE_SCRIPT), "--dry-run"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120,
        )
        assert res.returncode == 0, res.stderr
        assert "rotation_performed: True" in res.stdout
        assert "synth_op_" not in res.stdout
        assert "synth_op_" not in res.stderr
        # Real target file must not appear during dry-run.
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

    def test_no_db_in_sec06_report_dir(self):
        if not REPORT_DIR.exists():
            pytest.skip("report dir not yet created")
        for p in REPORT_DIR.rglob("*"):
            if p.is_file():
                assert p.suffix.lower() not in (".db", ".sqlite", ".sqlite3")


# ---------------------------------------------------------------------------
# TestPublicReport
# ---------------------------------------------------------------------------


class TestPublicReport:
    @pytest.fixture(scope="class")
    def report(self):
        from scripts.run_cka_sec06_key_rotation_validation import run_validation
        return run_validation()

    def test_block_id(self, report):
        assert report["block_id"] == "CKA-SEC-06"

    def test_conclusion(self, report):
        assert report["conclusion"] == "cka_sec06_key_rotation_plan_ready"

    def test_round_trip_flags(self, report):
        for k in (
            "key_rotation_tool_ready",
            "synthetic_rotation_rehearsal_passed",
            "backup_before_rotation_required",
            "backup_checksum_verified",
            "new_key_open_after_rotation_passed",
            "old_key_rejected_after_rotation",
            "record_count_preserved",
            "rollback_restore_verified",
            "plaintext_absence_verified",
            "command_line_keys_rejected",
            "empty_key_refused",
            "key_mismatch_refused",
            "same_old_new_key_refused",
            "real_rotation_blocked_by_default",
        ):
            assert report[k] is True, f"flag {k} not True"

    def test_safety_flags_false(self, report):
        for k in (
            "real_rotation_performed",
            "real_store_touched",
            "real_data_touched",
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
            "cka_sec06_src_", "cka_sec06_pre_rekey_bk_", "cka_sec06_rollback_tgt_",
            "cka_sec06_v_", "cka_sec06_pytest_",
            "cka_sec07_", "cka_sec05_", "cka_sec04_", "cka_sec03a_",
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
        md = REPORT_DIR / "cka_sec06_key_rotation_report.md"
        assert md.exists()

    def test_operator_guide_present(self):
        guide = REPORT_DIR / "CKA_SEC06_KEY_ROTATION_OPERATOR_GUIDE.md"
        assert guide.exists()
        text = guide.read_text(encoding="utf-8")
        assert "PRAGMA rekey" in text
        assert "getpass" in text


# ---------------------------------------------------------------------------
# TestRollbackPlan
# ---------------------------------------------------------------------------


class TestRollbackPlan:
    def test_plan_has_required_steps(self):
        steps = key_rotation_rollback_steps()
        assert isinstance(steps, list)
        assert len(steps) >= 5
        joined = "\n".join(steps)
        assert "Backup" in joined
        assert "verified" in joined.lower()
        assert "lost key" in joined.lower()


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
            "clinical_knowledge/security/key_rotation.py",
            "clinical_knowledge/security/key_rotation_plan.py",
            "scripts/cka_encrypted_store_rotate_key.py",
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
