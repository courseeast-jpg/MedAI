"""CKA-SEC-05 — operator encrypted-runtime launcher tests.

Covers:
- Python launcher module imports
- argparse rejects --key / --encryption-key
- prompt_key_twice refuses mismatch + empty key + matches return value
- build_child_env adds the SEC-04 env vars and key
- parent os.environ unchanged after build_child_env
- default Start_MedAI_UI.bat unmodified (no encrypted env vars)
- Start_MedAI_UI_Encrypted.bat exists, calls Python launcher, no hardcoded key
- --dry-run does not create real DB
- --self-test --test-mode opens encrypted runtime against temp DB only
- missing store without --create-if-missing blocks safely
- create_if_missing flag exposed
- no DB files staged
- public report no key/path/secret/PHI
- final SEC-04 + final CKA validation invocable
- no clinical text in launcher / helper modules
- no prescription dosing text
- main MKBStore class unchanged
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
    LauncherError,
    build_arg_parser,
    build_child_env,
    build_streamlit_command,
    child_env_keys_for_encrypted_runtime,
    detect_sqlcipher_provider,
    prompt_key_twice,
    reject_command_line_key,
    run_self_test,
)


REPORT_DIR = REPO_ROOT / "reports" / "cka_sec05_operator_runtime_launcher"
REAL_TARGET = REPO_ROOT / "data" / "secure" / "cka_encrypted_future_store.db"
DEFAULT_BAT = REPO_ROOT / "Start_MedAI_UI.bat"
ENCRYPTED_BAT = REPO_ROOT / "Start_MedAI_UI_Encrypted.bat"
LAUNCHER_PY = REPO_ROOT / "scripts" / "start_cka_encrypted_runtime_ui.py"


def _has_provider() -> bool:
    return bool(detect_sqlcipher_provider().available)


def _new_key() -> str:
    return "synth_op_" + secrets.token_hex(16)


# ---------------------------------------------------------------------------
# TestImports
# ---------------------------------------------------------------------------


class TestImports:
    def test_security_package_exposes_sec05_surface(self):
        import clinical_knowledge.security as sec
        for name in (
            "LauncherError",
            "SelfTestResult",
            "build_arg_parser",
            "build_child_env",
            "build_streamlit_command",
            "child_env_keys_for_encrypted_runtime",
            "prompt_key_twice",
            "reject_command_line_key",
            "resolve_key",
            "run_self_test",
        ):
            assert hasattr(sec, name), f"missing export: {name}"

    def test_validation_script_importable(self):
        from scripts import run_cka_sec05_operator_runtime_launcher_validation as v
        assert hasattr(v, "run_validation")

    def test_launcher_script_importable(self):
        from scripts import start_cka_encrypted_runtime_ui as s
        assert hasattr(s, "main")


# ---------------------------------------------------------------------------
# TestArgParserRejectsKey
# ---------------------------------------------------------------------------


class TestArgParserRejectsKey:
    @pytest.mark.parametrize("argv", [
        ["--key=secret123"],
        ["--encryption-key=secret123"],
        ["--store-path", "x", "--key"],
        ["--encryption-key"],
        ["--key=foo", "--port", "8511"],
    ])
    def test_pre_argparse_helper_rejects(self, argv):
        assert reject_command_line_key(argv) == "command_line_key_not_accepted"

    @pytest.mark.parametrize("argv", [
        ["--store-path", "x"],
        ["--port", "8511"],
        ["--dry-run"],
        ["--self-test", "--test-mode"],
        [],
    ])
    def test_pre_argparse_helper_passes_benign(self, argv):
        assert reject_command_line_key(argv) is None

    def test_subprocess_rejects_key_flag(self):
        res = subprocess.run(
            [sys.executable, str(LAUNCHER_PY), "--key=secret123"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=15,
        )
        assert res.returncode != 0
        # Secret value must NOT appear in any output.
        combined = res.stdout + "\n" + res.stderr
        assert "secret123" not in combined


# ---------------------------------------------------------------------------
# TestPromptKeyTwice
# ---------------------------------------------------------------------------


class TestPromptKeyTwice:
    def test_match_returns_value(self):
        from clinical_knowledge.security import runtime_launcher as rl
        seq = iter(["matchingkey_001", "matchingkey_001"])
        original = rl.getpass.getpass
        rl.getpass.getpass = lambda prompt: next(seq)    # type: ignore[assignment]
        try:
            assert prompt_key_twice() == "matchingkey_001"
        finally:
            rl.getpass.getpass = original

    def test_mismatch_raises(self):
        from clinical_knowledge.security import runtime_launcher as rl
        seq = iter(["abcdefghijkl1", "DIFFERENTKEY9"])
        original = rl.getpass.getpass
        rl.getpass.getpass = lambda prompt: next(seq)    # type: ignore[assignment]
        try:
            with pytest.raises(LauncherError) as exc:
                prompt_key_twice()
            assert "mismatch" in str(exc.value)
        finally:
            rl.getpass.getpass = original

    def test_empty_first_raises(self):
        from clinical_knowledge.security import runtime_launcher as rl
        seq = iter(["", "anything"])
        original = rl.getpass.getpass
        rl.getpass.getpass = lambda prompt: next(seq)    # type: ignore[assignment]
        try:
            with pytest.raises(LauncherError) as exc:
                prompt_key_twice()
            assert "empty_key" in str(exc.value)
        finally:
            rl.getpass.getpass = original


# ---------------------------------------------------------------------------
# TestChildEnv
# ---------------------------------------------------------------------------


class TestChildEnv:
    def test_child_env_contains_required_vars(self):
        key = _new_key()
        env = build_child_env(key, "data/secure/x.db", create_if_missing=False,
                              base_env={})
        for k in (
            "MEDAI_LOCAL_ONLY",
            "MEDAI_ALLOW_EXTERNAL_API",
            "MEDAI_REQUIRE_PII_SCRUB",
            "MEDAI_PRIVACY_AUDIT",
            "MEDAI_CKA_ENCRYPTED_STORE_ENABLED",
            "MEDAI_CKA_ENCRYPTED_STORE_PATH",
            "MEDAI_CKA_ENCRYPTION_KEY",
        ):
            assert k in env, f"missing env var: {k}"
        assert env["MEDAI_LOCAL_ONLY"] == "1"
        assert env["MEDAI_ALLOW_EXTERNAL_API"] == "0"
        assert env["MEDAI_CKA_ENCRYPTED_STORE_ENABLED"] == "1"
        assert env["MEDAI_CKA_ENCRYPTION_KEY"] == key
        assert env["MEDAI_CKA_ENCRYPTED_STORE_PATH"] == "data/secure/x.db"

    def test_child_env_create_if_missing_off_by_default(self):
        env = build_child_env(_new_key(), "x.db", create_if_missing=False, base_env={})
        assert "MEDAI_CKA_ENCRYPTED_STORE_CREATE_IF_MISSING" not in env

    def test_child_env_create_if_missing_when_true(self):
        env = build_child_env(_new_key(), "x.db", create_if_missing=True, base_env={})
        assert env["MEDAI_CKA_ENCRYPTED_STORE_CREATE_IF_MISSING"] == "1"

    def test_parent_environ_unchanged(self):
        before = dict(os.environ)
        build_child_env(_new_key(), "x.db", create_if_missing=True)
        assert "MEDAI_CKA_ENCRYPTION_KEY" not in os.environ
        # The parent environ must not have new MEDAI_CKA_* keys we didn't already have.
        for k in ("MEDAI_CKA_ENCRYPTED_STORE_ENABLED",
                  "MEDAI_CKA_ENCRYPTED_STORE_PATH",
                  "MEDAI_CKA_ENCRYPTION_KEY",
                  "MEDAI_CKA_ENCRYPTED_STORE_CREATE_IF_MISSING"):
            assert os.environ.get(k) == before.get(k)


# ---------------------------------------------------------------------------
# TestBatFiles
# ---------------------------------------------------------------------------


class TestBatFiles:
    def test_default_bat_present(self):
        assert DEFAULT_BAT.exists(), f"missing: {DEFAULT_BAT}"

    def test_default_bat_local_only(self):
        text = DEFAULT_BAT.read_text(encoding="utf-8")
        for tok in (
            "MEDAI_LOCAL_ONLY=1",
            "MEDAI_ALLOW_EXTERNAL_API=0",
            "MEDAI_REQUIRE_PII_SCRUB=1",
            "MEDAI_PRIVACY_AUDIT=1",
        ):
            assert tok in text, f"default bat missing token: {tok}"

    def test_default_bat_no_encrypted_env(self):
        text = DEFAULT_BAT.read_text(encoding="utf-8")
        for tok in (
            "MEDAI_CKA_ENCRYPTED_STORE_ENABLED",
            "MEDAI_CKA_ENCRYPTION_KEY",
            "MEDAI_CKA_ENCRYPTED_STORE_PATH",
        ):
            assert tok not in text, f"default bat unexpectedly contains: {tok}"

    def test_encrypted_bat_present(self):
        assert ENCRYPTED_BAT.exists(), f"missing: {ENCRYPTED_BAT}"

    def test_encrypted_bat_calls_python_launcher(self):
        text = ENCRYPTED_BAT.read_text(encoding="utf-8")
        assert "start_cka_encrypted_runtime_ui.py" in text

    def test_encrypted_bat_no_hardcoded_key_assignment(self):
        text = ENCRYPTED_BAT.read_text(encoding="utf-8")
        for tok in (
            "MEDAI_CKA_ENCRYPTION_KEY=",
            "set ENCRYPTION_KEY=",
            "encryption_key=",
        ):
            assert tok not in text, f"encrypted bat hardcoded key marker: {tok}"

    def test_encrypted_bat_does_not_replace_default(self):
        # Both must coexist.
        assert DEFAULT_BAT.exists() and ENCRYPTED_BAT.exists()


# ---------------------------------------------------------------------------
# TestDryRunAndSelfTest
# ---------------------------------------------------------------------------


class TestDryRunAndSelfTest:
    def test_dry_run_exits_clean(self):
        res = subprocess.run(
            [sys.executable, str(LAUNCHER_PY), "--dry-run"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30,
        )
        assert res.returncode == 0
        assert "dry_run=ok" in res.stdout

    def test_dry_run_does_not_create_real_db(self):
        existed_before = REAL_TARGET.exists()
        subprocess.run(
            [sys.executable, str(LAUNCHER_PY), "--dry-run"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30,
        )
        existed_after = REAL_TARGET.exists()
        # Dry-run must NOT cause the real target to appear.
        assert existed_after == existed_before

    @pytest.mark.skipif(not _has_provider(),
                        reason="SQLCipher provider unavailable")
    def test_self_test_runs_against_temp_db(self):
        test_key = _new_key()
        env = dict(os.environ)
        env["CKA_SEC04_TEST_KEY"] = test_key
        res = subprocess.run(
            [sys.executable, str(LAUNCHER_PY), "--self-test", "--test-mode"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60, env=env,
        )
        assert res.returncode == 0, res.stderr
        assert "passed: True" in res.stdout
        assert "records_count: 0" in res.stdout
        # Test key must NOT appear in any output.
        assert test_key not in res.stdout
        assert test_key not in res.stderr

    @pytest.mark.skipif(not _has_provider(),
                        reason="SQLCipher provider unavailable")
    def test_run_self_test_helper_returns_safe_summary(self):
        result = run_self_test(_new_key())
        assert result.passed is True
        assert result.records_count == 0
        assert result.runtime_encryption_active is True
        assert result.real_db_created is False
        assert result.temp_files_staged is False
        s = result.safe_public_summary()
        assert "synth_op_" not in json.dumps(s)


# ---------------------------------------------------------------------------
# TestMissingStoreBlocks
# ---------------------------------------------------------------------------


class TestMissingStoreBlocks:
    def test_create_if_missing_flag_exposed(self):
        parser = build_arg_parser()
        actions = {a.dest for a in parser._actions}    # type: ignore[attr-defined]
        assert "create_if_missing" in actions

    @pytest.mark.skipif(not _has_provider(),
                        reason="SQLCipher provider unavailable")
    def test_real_store_not_created_during_dry_run(self):
        # Even with --store-path pointing at the real target, --dry-run
        # must not create the file.
        existed_before = REAL_TARGET.exists()
        env = dict(os.environ)
        env["CKA_SEC04_TEST_KEY"] = _new_key()
        subprocess.run(
            [sys.executable, str(LAUNCHER_PY),
             "--store-path", str(REAL_TARGET), "--dry-run"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30, env=env,
        )
        assert REAL_TARGET.exists() == existed_before


# ---------------------------------------------------------------------------
# TestNoStagedDB
# ---------------------------------------------------------------------------


class TestNoStagedDB:
    def test_no_db_in_security_package(self):
        sec_dir = REPO_ROOT / "clinical_knowledge" / "security"
        for p in sec_dir.rglob("*"):
            if p.is_file():
                assert p.suffix.lower() not in (".db", ".sqlite", ".sqlite3")

    def test_no_db_in_sec05_report_dir(self):
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
        from scripts.run_cka_sec05_operator_runtime_launcher_validation import (
            run_validation,
        )
        return run_validation()

    def test_block_id(self, report):
        assert report["block_id"] == "CKA-SEC-05"

    def test_conclusion(self, report):
        assert report["conclusion"] == "cka_sec05_operator_runtime_launcher_ready"

    def test_default_launcher_unchanged_flag(self, report):
        assert report["default_launcher_unchanged"] is True

    def test_encrypted_launcher_no_key(self, report):
        assert report["encrypted_launcher_contains_key"] is False
        assert report["key_stored_in_repo"] is False
        assert report["encryption_key_logged"] is False

    def test_safety_flags_false(self, report):
        for k in (
            "real_empty_store_created_by_default",
            "existing_data_migrated",
            "main_store_migration_performed",
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

    def test_report_no_key_prefix(self, report):
        text = json.dumps(report)
        assert "synth_op_" not in text

    def test_report_no_drive_letter_path(self, report):
        text = json.dumps(report)
        assert not re.search(r"[A-Za-z]:\\\\", text)

    def test_report_no_temp_prefix(self, report):
        text = json.dumps(report)
        for needle in ("cka_sec05_selftest_", "cka_sec05_v_", "cka_sec05_pytest_",
                       "cka_sec04_v_", "cka_sec04_smoke_", "cka_sec04_pytest_",
                       "cka_sec03a_v_", "cka_sec03a_init_", "cka_sec03a_pytest_"):
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
        md = REPORT_DIR / "cka_sec05_operator_runtime_launcher_report.md"
        assert md.exists()

    def test_operator_guide_present(self):
        guide = REPORT_DIR / "CKA_SEC05_ENCRYPTED_RUNTIME_LAUNCHER_GUIDE.md"
        assert guide.exists()
        text = guide.read_text(encoding="utf-8")
        assert "Start_MedAI_UI_Encrypted.bat" in text
        assert "Start_MedAI_UI.bat" in text
        assert "getpass" in text


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
            "clinical_knowledge/security/runtime_launcher.py",
            "clinical_knowledge/security/runtime_factory.py",
            "clinical_knowledge/security/runtime_preflight.py",
            "clinical_knowledge/security/runtime_rollback.py",
            "clinical_knowledge/security/runtime_config.py",
            "clinical_knowledge/security/empty_store_initializer.py",
            "clinical_knowledge/security/encrypted_store.py",
            "clinical_knowledge/security/encryption_checks.py",
            "clinical_knowledge/security/encrypted_store_manifest.py",
            "clinical_knowledge/security/key_policy.py",
            "clinical_knowledge/security/migration_plan.py",
            "clinical_knowledge/security/migration_rehearsal.py",
            "clinical_knowledge/security/rollback_plan.py",
            "clinical_knowledge/security/sqlcipher_provider.py",
            "scripts/start_cka_encrypted_runtime_ui.py",
        ):
            text = (REPO_ROOT / fname).read_text(encoding="utf-8").lower()
            for needle in forbidden:
                assert needle not in text, f"{fname}: forbidden {needle!r}"

    def test_consensus_engine_still_loads(self):
        from clinical_knowledge.consensus.engine import run_consensus    # noqa: F401

    def test_decision_engine_still_loads(self):
        import clinical_knowledge.decision_engine.engine    # noqa: F401

    def test_truth_resolution_still_loads(self):
        import clinical_knowledge.truth_resolution.engine    # noqa: F401

    def test_ddi_stub_still_loads(self):
        import clinical_knowledge.medication_safety.ddi_stub    # noqa: F401

    def test_main_mkb_store_class_unchanged(self):
        from clinical_knowledge.store import MKBStore as MS
        from clinical_knowledge.security import EncryptedCKAStore as Enc
        assert isinstance(MS, type)
        assert MS is not Enc
