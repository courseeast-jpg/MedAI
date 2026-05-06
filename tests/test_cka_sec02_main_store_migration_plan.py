"""CKA-SEC-02 — Main store migration plan + synthetic rehearsal tests.

Covers:
- migration plan defaults to rehearsal_only
- real_migration_approved / main_store_migration_performed /
  real_data_migrated default False; True values raise
- key handling: empty / hardcoded keys rejected
- approval checklist generated on disk
- backup + rollback readiness
- store inventory returns safe hashes only (no raw paths)
- synthetic source DB creation works
- encrypted target DB creation works
- synthetic records copied
- correct-key read works
- wrong-key failure works
- plaintext absence verified
- source DB unchanged
- no temp DB staged
- validation script succeeds
- public reports contain no raw private strings
- no external API used
- no clinical recommendation text generated
- no prescription dosing text generated
- main MKB store class identity preserved
"""
from __future__ import annotations

import json
import os
import re
import secrets
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from clinical_knowledge.security import (    # noqa: E402
    BackupRollbackPolicy,
    EncryptedCKAStore,
    EncryptedStoreError,
    InventoryResult,
    KeyPolicyError,
    KeyPolicyStatus,
    MigrationPlan,
    MigrationRehearsalResult,
    backup_policy_ready,
    detect_sqlcipher_provider,
    get_backup_rollback_policy,
    get_key_policy_status,
    inventory_candidate_db_files,
    key_policy_ready,
    operator_approval_checklist,
    rehearsal_passed,
    rehearse_synthetic_migration,
    rollback_policy_ready,
    validate_operator_key,
)


REPORT_DIR = REPO_ROOT / "reports" / "cka_sec02_main_store_migration_plan"
CHECKLIST_NAME = "CKA_SEC02_OPERATOR_MIGRATION_APPROVAL_CHECKLIST.md"


def _has_provider() -> bool:
    return bool(detect_sqlcipher_provider().available)


def _new_synth_op_key() -> str:
    return "synth_op_" + secrets.token_hex(16)


# ---------------------------------------------------------------------------
# TestImports
# ---------------------------------------------------------------------------


class TestImports:
    def test_security_package_exposes_sec02_surface(self):
        import clinical_knowledge.security as sec
        for name in (
            "MigrationPlan",
            "MigrationRehearsalResult",
            "InventoryResult",
            "KeyPolicyError",
            "KeyPolicyStatus",
            "BackupRollbackPolicy",
            "rehearse_synthetic_migration",
            "rehearsal_passed",
            "inventory_candidate_db_files",
            "validate_operator_key",
            "operator_approval_checklist",
            "get_key_policy_status",
            "get_backup_rollback_policy",
            "key_policy_ready",
            "backup_policy_ready",
            "rollback_policy_ready",
        ):
            assert hasattr(sec, name), f"missing export: {name}"

    def test_validation_script_importable(self):
        from scripts import run_cka_sec02_main_store_migration_plan as v
        assert hasattr(v, "run_validation")


# ---------------------------------------------------------------------------
# TestMigrationPlanDefaults
# ---------------------------------------------------------------------------


class TestMigrationPlanDefaults:
    def test_defaults_to_rehearsal_only(self):
        plan = MigrationPlan.for_rehearsal(
            source_store_safe_id="cka_db_aaa",
            target_store_safe_id="cka_db_bbb",
            sqlcipher_provider="sqlcipher3",
            provider_version="4.x",
        )
        assert plan.migration_mode == "rehearsal_only"
        assert plan.real_migration_approved is False
        assert plan.main_store_migration_performed is False
        assert plan.real_data_migrated is False
        assert plan.backup_required is True
        assert plan.rollback_required is True
        assert plan.operator_approval_required is True

    def test_real_migration_approved_true_raises(self):
        with pytest.raises(ValueError):
            MigrationPlan(
                plan_id="x", created_at="x",
                source_store_safe_id="x", target_store_safe_id="y",
                sqlcipher_provider="sqlcipher3", provider_version="x",
                real_migration_approved=True,
            )

    def test_main_store_migration_performed_true_raises(self):
        with pytest.raises(ValueError):
            MigrationPlan(
                plan_id="x", created_at="x",
                source_store_safe_id="x", target_store_safe_id="y",
                sqlcipher_provider="sqlcipher3", provider_version="x",
                main_store_migration_performed=True,
            )

    def test_real_data_migrated_true_raises(self):
        with pytest.raises(ValueError):
            MigrationPlan(
                plan_id="x", created_at="x",
                source_store_safe_id="x", target_store_safe_id="y",
                sqlcipher_provider="sqlcipher3", provider_version="x",
                real_data_migrated=True,
            )

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            MigrationPlan(
                plan_id="x", created_at="x",
                source_store_safe_id="x", target_store_safe_id="y",
                sqlcipher_provider="sqlcipher3", provider_version="x",
                migration_mode="real_migration",
            )

    def test_safe_public_summary_no_key_field(self):
        plan = MigrationPlan.for_rehearsal(
            source_store_safe_id="cka_db_aaa",
            target_store_safe_id="cka_db_bbb",
            sqlcipher_provider="sqlcipher3",
            provider_version="4.x",
        )
        s = plan.safe_public_summary()
        text = json.dumps(s)
        assert "encryption_key" not in text
        assert "synth_op_" not in text


# ---------------------------------------------------------------------------
# TestKeyPolicy
# ---------------------------------------------------------------------------


class TestKeyPolicy:
    def test_status_summary_fields(self):
        s = get_key_policy_status().safe_public_summary()
        for k in (
            "no_hardcoded_key_in_code",
            "operator_provided_key_required",
            "key_logged_in_reports",
            "key_committed_to_git",
            "key_in_environment_dump",
            "confirm_twice_required_in_sec03",
            "recovery_warning_documented",
            "rotation_out_of_scope_for_sec02",
        ):
            assert k in s

    def test_policy_ready(self):
        assert key_policy_ready() is True

    def test_summary_has_no_real_key(self):
        s = get_key_policy_status().safe_public_summary()
        text = json.dumps(s)
        assert "synth_op_" not in text
        # Booleans only — no string with key-like length
        for v in s.values():
            assert isinstance(v, bool)

    @pytest.mark.parametrize("bad", [
        "", "x", "short", "shortish",
    ])
    def test_short_keys_refused(self, bad):
        with pytest.raises(KeyPolicyError):
            validate_operator_key(bad)

    @pytest.mark.parametrize("bad", [
        "your_key_here", "REPLACE_ME",
    ])
    def test_hardcoded_marker_refused(self, bad):
        with pytest.raises(KeyPolicyError):
            validate_operator_key(bad)

    def test_non_string_refused(self):
        with pytest.raises(KeyPolicyError):
            validate_operator_key(None)    # type: ignore[arg-type]
        with pytest.raises(KeyPolicyError):
            validate_operator_key(b"binary")    # type: ignore[arg-type]

    def test_synthetic_op_key_accepted(self):
        # Should NOT raise.
        validate_operator_key(_new_synth_op_key())

    def test_no_hardcoded_key_in_security_package(self):
        # The security package source must not contain a real-looking key.
        for fname in (
            "clinical_knowledge/security/key_policy.py",
            "clinical_knowledge/security/migration_plan.py",
            "clinical_knowledge/security/migration_rehearsal.py",
            "clinical_knowledge/security/encrypted_store.py",
            "clinical_knowledge/security/sqlcipher_provider.py",
            "clinical_knowledge/security/encryption_checks.py",
            "clinical_knowledge/security/rollback_plan.py",
        ):
            text = (REPO_ROOT / fname).read_text(encoding="utf-8")
            # synth_op_ is only generated at runtime via secrets.token_hex
            for needle in ('PRAGMA key = "', "encryption_key = '", 'PASSWORD = "'):
                assert needle not in text, f"{fname} contains hardcoded key"

    def test_approval_checklist_has_six_items(self):
        items = operator_approval_checklist()
        assert isinstance(items, list)
        assert len(items) == 6
        # All items are non-empty strings.
        for item in items:
            assert isinstance(item, str) and len(item) > 5


# ---------------------------------------------------------------------------
# TestBackupRollbackPolicy
# ---------------------------------------------------------------------------


class TestBackupRollbackPolicy:
    def test_backup_ready(self):
        assert backup_policy_ready() is True

    def test_rollback_ready(self):
        assert rollback_policy_ready() is True

    def test_real_backup_not_performed(self):
        p = get_backup_rollback_policy()
        assert p.real_backup_performed is False
        assert p.real_restore_performed is False
        assert p.real_db_deleted is False

    def test_rollback_steps_present(self):
        p = get_backup_rollback_policy()
        assert len(p.rollback_steps) >= 5

    def test_summary_no_real_paths(self):
        s = get_backup_rollback_policy().safe_public_summary()
        text = json.dumps(s)
        assert not re.search(r"[A-Za-z]:\\\\", text)
        assert "rollback_steps" not in s   # only count, no raw text in summary

    def test_approval_checklist_file_exists(self):
        path = REPORT_DIR / CHECKLIST_NAME
        assert path.exists(), f"missing checklist file: {path.name}"
        text = path.read_text(encoding="utf-8").lower()
        assert "operator confirms" in text
        assert "lost key" in text


# ---------------------------------------------------------------------------
# TestInventory
# ---------------------------------------------------------------------------


class TestInventory:
    def test_inventory_returns_safe_hashes_only(self):
        inv = inventory_candidate_db_files()
        s = inv.safe_public_summary()
        for h in s["candidate_db_safe_hashes"]:
            assert isinstance(h, str)
            assert h.startswith("cka_db_")
            # Each hash is exactly the prefix + 16 hex chars.
            assert len(h) == len("cka_db_") + 16

    def test_inventory_no_raw_paths_in_summary(self):
        inv = inventory_candidate_db_files()
        text = json.dumps(inv.safe_public_summary())
        assert not re.search(r"[A-Za-z]:\\\\", text)
        # Forward-slash absolute path
        assert not re.search(r'"/[a-zA-Z]', text)
        # Repo-relative paths are confined to the safe scan_dirs list.
        assert "scan_dirs_relative" in inv.safe_public_summary()

    def test_inventory_does_not_modify_files(self, tmp_path):
        # Create a sentinel db file under data/ and confirm inventory
        # does not change its mtime/size.
        marker_dir = REPO_ROOT / "data"
        if not marker_dir.exists():
            pytest.skip("no data/ directory present")
        candidate = marker_dir / "_sec02_inventory_test_sentinel.db"
        candidate.write_bytes(b"sentinel")
        try:
            stat_before = candidate.stat()
            _ = inventory_candidate_db_files()
            stat_after = candidate.stat()
            assert stat_before.st_size == stat_after.st_size
            assert stat_before.st_mtime == stat_after.st_mtime
        finally:
            candidate.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# TestSyntheticRehearsalLive (requires SQLCipher provider)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_provider(),
                    reason="SQLCipher provider unavailable — rehearsal not exercised")
class TestSyntheticRehearsalLive:
    def test_rehearsal_end_to_end(self):
        result = rehearse_synthetic_migration(_new_synth_op_key())
        assert result.rehearsal_performed is True
        assert result.synthetic_source_created is True
        assert result.encrypted_target_created is True
        assert result.records_copied >= 1
        assert result.correct_key_read_passed is True
        assert result.wrong_key_failed is True
        assert result.plaintext_absence_verified is True
        assert result.source_unchanged is True
        assert result.temp_files_staged is False
        assert rehearsal_passed(result) is True

    def test_rehearsal_summary_no_path_no_key(self):
        result = rehearse_synthetic_migration(_new_synth_op_key())
        s = result.safe_public_summary()
        text = json.dumps(s)
        assert "synth_op_" not in text
        assert "cka_sec02_src_" not in text
        assert "cka_sec02_tgt_" not in text
        assert not re.search(r"[A-Za-z]:\\\\", text)

    def test_rehearsal_refuses_empty_key(self):
        with pytest.raises(KeyPolicyError):
            rehearse_synthetic_migration("")

    def test_rehearsal_refuses_short_key(self):
        with pytest.raises(KeyPolicyError):
            rehearse_synthetic_migration("short")

    def test_rehearsal_does_not_stage_temp_files(self):
        before = {p.name for p in REPORT_DIR.glob("*") if p.is_file()}
        _ = rehearse_synthetic_migration(_new_synth_op_key())
        after = {p.name for p in REPORT_DIR.glob("*") if p.is_file()}
        # No new files in the public report dir, and certainly no .db.
        assert all(not n.endswith(".db") for n in after)
        # The set of files in the report dir should not have grown with .db
        # contributions from the rehearsal.
        assert {n for n in after if n.endswith(".db")} == set()
        assert {n for n in before if n.endswith(".db")} == set()


# ---------------------------------------------------------------------------
# TestProviderUnavailableSkip
# ---------------------------------------------------------------------------


class TestRehearsalNoProvider:
    def test_rehearsal_raises_when_provider_unavailable(self):
        if _has_provider():
            pytest.skip("provider available")
        with pytest.raises(EncryptedStoreError):
            rehearse_synthetic_migration(_new_synth_op_key())


# ---------------------------------------------------------------------------
# TestMainStoreUntouched
# ---------------------------------------------------------------------------


class TestMainStoreUntouched:
    def test_mkb_store_class_identity_unchanged(self):
        from clinical_knowledge.store import MKBStore
        from clinical_knowledge.security import EncryptedCKAStore as Enc
        assert isinstance(MKBStore, type)
        assert MKBStore is not Enc

    def test_security_package_does_not_export_mkb_store(self):
        import clinical_knowledge.security as sec
        assert "MKBStore" not in dir(sec)

    def test_existing_cka_modules_still_load(self):
        import clinical_knowledge.preflight    # noqa: F401
        import clinical_knowledge.scaffold     # noqa: F401
        import clinical_knowledge.consensus.engine    # noqa: F401
        import app.clinical_knowledge_safety_viewer    # noqa: F401


# ---------------------------------------------------------------------------
# TestPublicReport
# ---------------------------------------------------------------------------


class TestPublicReport:
    @pytest.fixture(scope="class")
    def report(self):
        from scripts.run_cka_sec02_main_store_migration_plan import run_validation
        return run_validation()

    def test_block_id(self, report):
        assert report["block_id"] == "CKA-SEC-02"

    def test_conclusion(self, report):
        assert report["conclusion"] == "cka_sec02_main_store_migration_plan_ready"

    def test_main_store_flags_false(self, report):
        for k in (
            "real_migration_approved",
            "main_store_migration_performed",
            "real_data_migrated",
            "sqlcipher_encryption_active_for_main_store",
            "temp_db_files_staged",
        ):
            assert report[k] is False, f"flag {k} not False"

    def test_safety_flags_false(self, report):
        for k in (
            "external_api_used",
            "raw_phi_logged_in_public_reports",
            "encryption_key_logged",
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

    def test_report_no_op_key_prefix(self, report):
        assert "synth_op_" not in json.dumps(report)

    def test_report_no_drive_letter_path(self, report):
        assert not re.search(r"[A-Za-z]:\\\\", json.dumps(report))

    def test_report_no_temp_db_prefix(self, report):
        text = json.dumps(report)
        assert "cka_sec02_src_" not in text
        assert "cka_sec02_tgt_" not in text

    def test_report_passes_b02_privacy_checker(self, report):
        from clinical_knowledge.privacy.report_privacy import (
            check_public_report_payload,
        )
        result = check_public_report_payload(report)
        assert result.passed, (
            f"privacy checker rejected report: "
            f"{result.leak_examples_redacted}"
        )

    def test_inventory_summary_has_safe_hashes(self, report):
        inv = report["inventory_summary"]
        for h in inv["candidate_db_safe_hashes"]:
            assert h.startswith("cka_db_")

    def test_rehearsal_block_present(self, report):
        rs = report["rehearsal_summary"]
        for k in (
            "rehearsal_performed",
            "synthetic_source_created",
            "encrypted_target_created",
            "records_copied",
            "correct_key_read_passed",
            "wrong_key_failed",
            "plaintext_absence_verified",
            "source_unchanged",
            "temp_files_staged",
        ):
            assert k in rs

    def test_next_recommended_block_mentions_sec03(self, report):
        nxt = report.get("next_recommended_block", "")
        assert "SEC-03" in nxt

    def test_report_md_present(self):
        md = REPORT_DIR / "cka_sec02_main_store_migration_plan_report.md"
        assert md.exists()


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
