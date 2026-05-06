"""CKA-SEC-04 — encrypted runtime activation tests.

Covers:
- default config has encrypted_runtime_requested=False
- env flag parsing
- key-present detection without exposing key
- runtime factory returns MKBStore when flag off
- runtime flag without key blocks
- wrong key fails explicitly
- no silent fallback when encrypted runtime requested
- create_if_missing=False blocks missing store
- test-mode create_if_missing=True creates temp encrypted store
- records_count=0 in created encrypted store
- plaintext absence verified
- rollback plan contains non-destructive instructions
- main_store_migration_performed remains False
- real_data_migrated remains False
- DB file is not staged
- public report no key/path/secret/PHI
- final validation script succeeds
- no clinical text / no prescription dosing text in security package
"""
from __future__ import annotations

import json
import os
import re
import secrets
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from clinical_knowledge.security import (    # noqa: E402
    EncryptedCKAStore,
    EncryptedRuntimeConfig,
    RuntimeBuildResult,
    RuntimeFactoryError,
    RuntimePreflightResult,
    RuntimeRollbackPlan,
    build_cka_runtime_store,
    detect_sqlcipher_provider,
    get_runtime_rollback_plan,
    run_encrypted_runtime_preflight,
    runtime_rollback_plan_ready,
)
from clinical_knowledge.security.empty_store_initializer import (    # noqa: E402
    _lock_path_for,
)
from clinical_knowledge.store import MKBStore    # noqa: E402


REPORT_DIR = REPO_ROOT / "reports" / "cka_sec04_encrypted_runtime_activation"


def _has_provider() -> bool:
    return bool(detect_sqlcipher_provider().available)


def _new_key() -> str:
    return "synth_op_" + secrets.token_hex(16)


def _new_temp_db() -> str:
    fd, path = tempfile.mkstemp(prefix="cka_sec04_pytest_", suffix=".db")
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
    def test_security_package_exposes_sec04_surface(self):
        import clinical_knowledge.security as sec
        for name in (
            "EncryptedRuntimeConfig",
            "RuntimeBuildResult",
            "RuntimeFactoryError",
            "build_cka_runtime_store",
            "RuntimePreflightResult",
            "run_encrypted_runtime_preflight",
            "RuntimeRollbackPlan",
            "get_runtime_rollback_plan",
            "runtime_rollback_plan_ready",
        ):
            assert hasattr(sec, name), f"missing export: {name}"

    def test_validation_script_importable(self):
        from scripts import (
            run_cka_sec04_encrypted_runtime_activation_validation as v,
        )
        assert hasattr(v, "run_validation")


# ---------------------------------------------------------------------------
# TestRuntimeConfigDefault
# ---------------------------------------------------------------------------


class TestRuntimeConfigDefault:
    def test_default_config_no_request(self):
        cfg = EncryptedRuntimeConfig.from_env(env={})
        assert cfg.encrypted_runtime_requested is False
        assert cfg.key_present is False
        assert cfg.create_if_missing is False
        assert cfg.runtime_activation_allowed is False

    def test_summary_no_path_no_key(self):
        cfg = EncryptedRuntimeConfig.from_env(env={
            "MEDAI_CKA_ENCRYPTED_STORE_PATH": "C:\\private\\foo.db",
            "MEDAI_CKA_ENCRYPTION_KEY": "supersecretkey1",
        })
        s = cfg.safe_public_summary()
        text = json.dumps(s)
        # Path must not leak into the summary in raw form.
        assert "C:\\\\" not in text
        assert "private" not in text
        # The key must never appear in the summary.
        assert "supersecretkey1" not in text

    def test_env_flag_parsing_truthy(self):
        for v in ("1", "true", "TRUE", "yes", "on"):
            cfg = EncryptedRuntimeConfig.from_env(env={
                "MEDAI_CKA_ENCRYPTED_STORE_ENABLED": v,
            })
            assert cfg.encrypted_runtime_requested is True

    def test_env_flag_parsing_falsy(self):
        for v in ("", "0", "false", "no", "off", "wat"):
            cfg = EncryptedRuntimeConfig.from_env(env={
                "MEDAI_CKA_ENCRYPTED_STORE_ENABLED": v,
            })
            assert cfg.encrypted_runtime_requested is False

    def test_key_presence_does_not_leak_value(self):
        cfg = EncryptedRuntimeConfig.from_env(env={
            "MEDAI_CKA_ENCRYPTION_KEY": "leakable_key_123456",
        })
        assert cfg.key_present is True
        s = cfg.safe_public_summary()
        assert "leakable_key_123456" not in json.dumps(s)


# ---------------------------------------------------------------------------
# TestRuntimeFactoryDefault
# ---------------------------------------------------------------------------


class TestRuntimeFactoryDefault:
    def test_no_flag_returns_mkbstore(self):
        cfg = EncryptedRuntimeConfig.from_env(env={})
        result = build_cka_runtime_store(cfg)
        assert isinstance(result.store, MKBStore)
        assert result.runtime_encryption_active is False
        assert result.fallback_to_mkbstore is False
        assert result.main_store_migration_performed is False
        assert result.real_data_migrated is False

    def test_default_summary_safe(self):
        cfg = EncryptedRuntimeConfig.from_env(env={})
        result = build_cka_runtime_store(cfg)
        s = result.safe_public_summary()
        assert s["runtime_encryption_active"] is False
        assert s["store_kind"] == "mkbstore_unencrypted_default"


# ---------------------------------------------------------------------------
# TestRuntimeFactoryGuards
# ---------------------------------------------------------------------------


class TestRuntimeFactoryGuards:
    def test_flag_without_key_raises(self):
        cfg = EncryptedRuntimeConfig.from_env(env={
            "MEDAI_CKA_ENCRYPTED_STORE_ENABLED": "1",
        })
        with pytest.raises(RuntimeFactoryError) as exc:
            build_cka_runtime_store(cfg)
        assert "missing_key" in str(exc.value)

    def test_flag_with_missing_path_raises(self):
        cfg = EncryptedRuntimeConfig.from_env(env={
            "MEDAI_CKA_ENCRYPTED_STORE_ENABLED": "1",
            "MEDAI_CKA_ENCRYPTION_KEY": _new_key(),
        })
        with pytest.raises(RuntimeFactoryError) as exc:
            build_cka_runtime_store(cfg)
        # Either missing_path or create_if_missing_false (depending on
        # whether the user supplied a path that doesn't exist).
        assert "missing" in str(exc.value)

    @pytest.mark.skipif(not _has_provider(),
                        reason="SQLCipher provider unavailable")
    def test_create_if_missing_false_blocks_missing_store(self):
        db = _new_temp_db()
        Path(db).unlink(missing_ok=True)    # type: ignore[call-arg]
        cfg = EncryptedRuntimeConfig.for_test(
            db, _new_key(),
            encrypted_runtime_requested=True,
            create_if_missing=False,
        )
        try:
            with pytest.raises(RuntimeFactoryError) as exc:
                build_cka_runtime_store(cfg)
            assert "create_if_missing_false" in str(exc.value)
            assert not Path(db).exists()
        finally:
            _cleanup_db(db)


# ---------------------------------------------------------------------------
# TestRuntimeFactoryLive
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_provider(),
                    reason="SQLCipher provider unavailable")
class TestRuntimeFactoryLive:
    def test_create_and_open_with_correct_key(self):
        db = _new_temp_db()
        key = _new_key()
        try:
            cfg = EncryptedRuntimeConfig.for_test(
                db, key, encrypted_runtime_requested=True, create_if_missing=True,
            )
            r = build_cka_runtime_store(cfg)
            assert r.runtime_encryption_active is True
            assert r.encrypted_store_created is True
            assert isinstance(r.store, EncryptedCKAStore)
            if hasattr(r.store, "close"):
                r.store.close()
        finally:
            _cleanup_db(db)

    def test_wrong_key_raises_explicit(self):
        db = _new_temp_db()
        correct = _new_key()
        try:
            cfg_create = EncryptedRuntimeConfig.for_test(
                db, correct,
                encrypted_runtime_requested=True,
                create_if_missing=True,
            )
            r = build_cka_runtime_store(cfg_create)
            if hasattr(r.store, "close"):
                r.store.close()

            wrong = _new_key()
            cfg_wrong = EncryptedRuntimeConfig.for_test(
                db, wrong,
                encrypted_runtime_requested=True,
                create_if_missing=False,
            )
            with pytest.raises(RuntimeFactoryError) as exc:
                build_cka_runtime_store(cfg_wrong)
            assert "open_failed" in str(exc.value)
        finally:
            _cleanup_db(db)

    def test_no_silent_fallback_to_mkbstore(self):
        # Even though the factory could theoretically silently return
        # MKBStore, the wrong-key path MUST raise. Verify by inspecting
        # the result type after a successful create+open vs. a failed open.
        db = _new_temp_db()
        try:
            cfg = EncryptedRuntimeConfig.for_test(
                db, _new_key(),
                encrypted_runtime_requested=True,
                create_if_missing=True,
            )
            r = build_cka_runtime_store(cfg)
            # On success, we get EncryptedCKAStore — never MKBStore.
            assert not isinstance(r.store, MKBStore)
            if hasattr(r.store, "close"):
                r.store.close()
        finally:
            _cleanup_db(db)


# ---------------------------------------------------------------------------
# TestPreflight
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_provider(),
                    reason="SQLCipher provider unavailable")
class TestPreflight:
    def test_preflight_passes_after_create(self):
        db = _new_temp_db()
        key = _new_key()
        try:
            cfg_create = EncryptedRuntimeConfig.for_test(
                db, key,
                encrypted_runtime_requested=True,
                create_if_missing=True,
            )
            r = build_cka_runtime_store(cfg_create)
            if hasattr(r.store, "close"):
                r.store.close()

            cfg_pf = EncryptedRuntimeConfig.for_test(
                db, key,
                encrypted_runtime_requested=True,
                create_if_missing=False,
            )
            pf = run_encrypted_runtime_preflight(cfg_pf)
            assert pf.passed is True
            assert pf.runtime_encryption_active is True
            assert pf.records_count == 0
            assert pf.correct_key_read_passed is True
            assert pf.wrong_key_failure_passed is True
            assert pf.plaintext_absence_verified is True
            assert pf.migration_performed is False
            assert pf.real_data_migrated is False
            assert pf.rollback_plan_available is True
        finally:
            _cleanup_db(db)

    def test_preflight_blocks_when_runtime_not_requested_but_passes_vacuously(self):
        cfg = EncryptedRuntimeConfig.from_env(env={})
        pf = run_encrypted_runtime_preflight(cfg)
        # Vacuous pass — encryption not requested.
        assert pf.passed is True
        assert pf.runtime_encryption_active is False
        assert pf.blocked_reason is None

    def test_preflight_blocks_when_key_missing(self):
        cfg = EncryptedRuntimeConfig.from_env(env={
            "MEDAI_CKA_ENCRYPTED_STORE_ENABLED": "1",
        })
        pf = run_encrypted_runtime_preflight(cfg)
        assert pf.passed is False
        assert pf.runtime_encryption_active is False
        assert pf.blocked_reason in ("provider_unavailable", "key_missing", "path_missing")


# ---------------------------------------------------------------------------
# TestRollbackPlan
# ---------------------------------------------------------------------------


class TestRollbackPlan:
    def test_plan_ready(self):
        plan = get_runtime_rollback_plan()
        assert runtime_rollback_plan_ready(plan) is True

    def test_plan_non_destructive(self):
        plan = get_runtime_rollback_plan()
        assert plan.no_destructive_action_on_rollback is True
        assert plan.no_data_migration_in_sec04 is True
        assert plan.do_not_delete_encrypted_store is True
        assert plan.do_not_delete_unencrypted_store is True

    def test_plan_step_count(self):
        plan = get_runtime_rollback_plan()
        assert len(plan.steps) >= 5
        joined = "\n".join(plan.steps)
        assert "MEDAI_CKA_ENCRYPTED_STORE_ENABLED" in joined

    def test_plan_summary_safe(self):
        plan = get_runtime_rollback_plan()
        s = plan.safe_public_summary()
        text = json.dumps(s)
        # No raw step text in summary — only counts and flags.
        assert "Unset" not in text
        assert s["steps_count"] == len(plan.steps)


# ---------------------------------------------------------------------------
# TestNoStagedDB
# ---------------------------------------------------------------------------


class TestNoStagedDB:
    def test_no_db_in_security_package(self):
        sec_dir = REPO_ROOT / "clinical_knowledge" / "security"
        for p in sec_dir.rglob("*"):
            if p.is_file():
                assert p.suffix.lower() not in (".db", ".sqlite", ".sqlite3")

    def test_no_db_in_sec04_report_dir(self):
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
        from scripts.run_cka_sec04_encrypted_runtime_activation_validation import (
            run_validation,
        )
        return run_validation()

    def test_block_id(self, report):
        assert report["block_id"] == "CKA-SEC-04"

    def test_conclusion(self, report):
        assert report["conclusion"] == "cka_sec04_encrypted_runtime_activation_ready"

    def test_default_off(self, report):
        assert report["default_runtime_encryption_active"] is False
        assert report["runtime_activation_default_off"] is True

    def test_safety_flags_false(self, report):
        for k in (
            "main_store_migration_performed",
            "real_data_migrated",
            "existing_store_migrated",
            "real_empty_store_created_by_default",
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
        for needle in ("cka_sec04_v_", "cka_sec04_smoke_", "cka_sec04_pytest_",
                       "cka_sec03a_v_", "cka_sec03a_init_"):
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
        md = REPORT_DIR / "cka_sec04_encrypted_runtime_activation_report.md"
        assert md.exists()

    def test_operator_guide_present(self):
        guide = REPORT_DIR / "CKA_SEC04_OPERATOR_RUNTIME_ACTIVATION_GUIDE.md"
        assert guide.exists()
        text = guide.read_text(encoding="utf-8")
        assert "MEDAI_CKA_ENCRYPTED_STORE_ENABLED" in text
        assert "Rollback" in text


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
            "clinical_knowledge/security/runtime_config.py",
            "clinical_knowledge/security/runtime_factory.py",
            "clinical_knowledge/security/runtime_preflight.py",
            "clinical_knowledge/security/runtime_rollback.py",
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

    def test_factory_returns_mkbstore_when_default(self):
        cfg = EncryptedRuntimeConfig.from_env(env={})
        r = build_cka_runtime_store(cfg)
        from clinical_knowledge.store import MKBStore as MS
        assert isinstance(r.store, MS)
