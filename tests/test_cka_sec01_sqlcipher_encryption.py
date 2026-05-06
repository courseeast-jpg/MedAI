"""CKA-SEC-01 — SQLCipher encryption readiness tests.

Covers:
- provider detection returns a safe status
- unavailable provider does not crash
- empty key refused
- encryption key never appears in safe_public_summary
- adapter importable
- synthetic DB create/read works (only if provider available)
- wrong key fails (only if provider available)
- plaintext absence (only if provider available)
- provider-unavailable path does NOT fake success
- public report has no key/path/secret/PHI
- main store migration flag remains False
- main CKA store class identity unchanged
- final CKA validation can be invoked from validation script
- no clinical text generated
- no external API used
"""
from __future__ import annotations

import json
import os
import re
import secrets
import sys
import tempfile
from pathlib import Path
from typing import List

import pytest

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from clinical_knowledge.security import (    # noqa: E402
    EncryptedCKAStore,
    EncryptedStoreError,
    SQLCipherProviderStatus,
    detect_sqlcipher_provider,
    verify_cipher_version,
    verify_plaintext_absent,
    verify_wrong_key_fails,
)
from clinical_knowledge.security.encryption_checks import (    # noqa: E402
    SYNTHETIC_FORBIDDEN_STRINGS,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def provider_status():
    return detect_sqlcipher_provider()


def _has_provider() -> bool:
    return bool(detect_sqlcipher_provider().available)


def _new_synth_key() -> str:
    return "synthkey_" + secrets.token_hex(16)


def _temp_db() -> str:
    fd, path = tempfile.mkstemp(prefix="cka_sec01_test_", suffix=".db")
    os.close(fd)
    try:
        os.unlink(path)
    except OSError:
        pass
    return path


# ---------------------------------------------------------------------------
# TestPackageImports
# ---------------------------------------------------------------------------

class TestPackageImports:
    def test_security_package_importable(self):
        import clinical_knowledge.security as sec
        assert sec is not None

    def test_provider_module_importable(self):
        import clinical_knowledge.security.sqlcipher_provider as mod
        assert hasattr(mod, "detect_sqlcipher_provider")
        assert hasattr(mod, "SQLCipherProviderStatus")

    def test_encrypted_store_module_importable(self):
        import clinical_knowledge.security.encrypted_store as mod
        assert hasattr(mod, "EncryptedCKAStore")
        assert hasattr(mod, "EncryptedStoreError")

    def test_checks_module_importable(self):
        import clinical_knowledge.security.encryption_checks as mod
        assert hasattr(mod, "verify_wrong_key_fails")
        assert hasattr(mod, "verify_plaintext_absent")
        assert hasattr(mod, "verify_cipher_version")

    def test_validation_script_importable(self):
        from scripts import run_cka_sec01_sqlcipher_encryption_validation as mod
        assert hasattr(mod, "run_validation")


# ---------------------------------------------------------------------------
# TestProviderDetection
# ---------------------------------------------------------------------------

class TestProviderDetection:
    def test_returns_status_object(self, provider_status):
        assert isinstance(provider_status, SQLCipherProviderStatus)

    def test_has_required_fields(self, provider_status):
        s = provider_status.safe_public_summary()
        for k in ("provider_name", "available", "cipher_version",
                  "import_error_safe", "notes"):
            assert k in s

    def test_does_not_crash_on_missing_provider(self):
        # Calling repeatedly must not raise even if no provider is present.
        for _ in range(3):
            _ = detect_sqlcipher_provider()

    def test_summary_no_path_or_module_object(self, provider_status):
        s = provider_status.safe_public_summary()
        # No attribute named "module" must leak into the summary.
        assert "module" not in s
        # No drive-letter paths.
        for v in s.values():
            if isinstance(v, str):
                assert not re.search(r"[A-Za-z]:\\", v)
                assert "/" not in v or v == ""

    def test_available_consistent_with_cipher_version(self, provider_status):
        s = provider_status.safe_public_summary()
        if s["available"] is True:
            assert s["cipher_version"] is not None
        # The reverse implication is not required (an imported module may
        # still fail the cipher-version probe).


# ---------------------------------------------------------------------------
# TestEncryptedStoreAdapter
# ---------------------------------------------------------------------------

class TestEncryptedStoreAdapter:
    def test_empty_key_refused_in_memory(self):
        with pytest.raises(EncryptedStoreError) as exc_info:
            EncryptedCKAStore(":memory:", "")
        assert str(exc_info.value) == "empty_encryption_key_refused"

    def test_empty_key_refused_disk_path(self):
        with pytest.raises(EncryptedStoreError) as exc_info:
            EncryptedCKAStore("anything.db", "")
        assert str(exc_info.value) == "empty_encryption_key_refused"

    def test_non_string_key_refused(self):
        with pytest.raises(EncryptedStoreError):
            EncryptedCKAStore(":memory:", None)    # type: ignore[arg-type]

    def test_empty_db_path_refused(self):
        with pytest.raises(EncryptedStoreError):
            EncryptedCKAStore("", _new_synth_key())

    def test_provider_unavailable_raises(self):
        if _has_provider():
            pytest.skip("provider available — provider_unavailable path not exercised")
        with pytest.raises(EncryptedStoreError) as exc_info:
            EncryptedCKAStore(":memory:", _new_synth_key())
        assert str(exc_info.value) == "provider_unavailable"


@pytest.mark.skipif(not _has_provider(),
                    reason="no SQLCipher provider — encrypted operations not exercised")
class TestEncryptedStoreLive:
    """Only exercised when a real SQLCipher provider is present."""

    def test_create_insert_read_with_correct_key(self):
        db = _temp_db()
        key = _new_synth_key()
        try:
            with EncryptedCKAStore(db, key) as store:
                store.insert_synthetic_record("r1", "label_alpha",
                                              "SYNTHETIC_PRIVATE_NAME_ALPHA")
                rows = store.fetch_all_synthetic()
            assert len(rows) == 1
            assert rows[0][0] == "r1"
        finally:
            Path(db).unlink(missing_ok=True)

    def test_wrong_key_fails(self):
        db = _temp_db()
        key = _new_synth_key()
        try:
            with EncryptedCKAStore(db, key) as store:
                store.insert_synthetic_record("r1", "label_alpha",
                                              "SYNTHETIC_PRIVATE_NAME_ALPHA")
            wrong = key + "_wrong"
            assert verify_wrong_key_fails(db, wrong) is True
        finally:
            Path(db).unlink(missing_ok=True)

    def test_plaintext_absent(self):
        db = _temp_db()
        key = _new_synth_key()
        try:
            with EncryptedCKAStore(db, key) as store:
                store.insert_synthetic_record(
                    "r1", "label_alpha",
                    "SYNTHETIC_PRIVATE_NAME_ALPHA SYNTHETIC_MRN_0001",
                )
            assert verify_plaintext_absent(db, SYNTHETIC_FORBIDDEN_STRINGS) is True
        finally:
            Path(db).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# TestKeyHandling — key must NEVER appear in summaries
# ---------------------------------------------------------------------------

class TestKeyHandling:
    def test_provider_summary_has_no_key_field(self, provider_status):
        s = provider_status.safe_public_summary()
        for k in s.keys():
            assert "key" not in k.lower() or k == "encryption_key_logged".lower() and False
        # Negative phrasing: the literal field "encryption_key" is absent.
        assert "encryption_key" not in s

    def test_provider_summary_no_key_value(self, provider_status):
        s = provider_status.safe_public_summary()
        for v in s.values():
            if isinstance(v, str):
                assert "synthkey_" not in v

    def test_adapter_summary_excludes_key_when_provider_available(self):
        if not _has_provider():
            pytest.skip("provider unavailable")
        db = _temp_db()
        key = _new_synth_key()
        try:
            with EncryptedCKAStore(db, key) as store:
                summary = store.safe_public_summary()
            text = json.dumps(summary)
            assert key not in text
            assert "synthkey_" not in text
            assert summary.get("encryption_key_logged") is False
            assert summary.get("encryption_key_in_summary") is False
        finally:
            Path(db).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# TestProviderUnavailablePathDoesNotFakeSuccess
# ---------------------------------------------------------------------------

class TestProviderUnavailableHonest:
    """When no provider is present, validation must NOT report success
    on the encryption operations."""

    def test_validation_marks_skipped_when_provider_unavailable(self):
        if _has_provider():
            pytest.skip("provider available — unavailable-path not testable")
        from scripts.run_cka_sec01_sqlcipher_encryption_validation import run_validation
        report = run_validation()
        # Honest conclusion when no provider is present
        assert report["conclusion"] == "cka_sec01_sqlcipher_provider_required"
        # These three encryption flags MUST be False when provider is absent
        assert report["synthetic_encrypted_store_created"] is False
        assert report["correct_key_read_passed"] is False
        assert report["wrong_key_read_failed"] is False
        assert report["plaintext_absence_verified"] is False
        assert report["sqlcipher_encryption_active_for_main_store"] is False
        # Cases C, D, E must be marked skipped
        for case in report["case_results"]:
            if case["case"] in ("C", "D", "E"):
                assert case.get("skipped") is True
                assert case.get("skip_reason") == "skipped_provider_unavailable"


# ---------------------------------------------------------------------------
# TestMainStoreUntouched
# ---------------------------------------------------------------------------

class TestMainStoreUntouched:
    def test_mkb_store_class_unchanged(self):
        from clinical_knowledge.store import MKBStore
        assert isinstance(MKBStore, type)
        assert MKBStore is not EncryptedCKAStore

    def test_security_package_does_not_replace_mkb_store(self):
        import clinical_knowledge.security as sec
        from clinical_knowledge.store import MKBStore
        # Security package exposes EncryptedCKAStore but not MKBStore.
        assert "MKBStore" not in dir(sec)
        # Original MKBStore identity intact.
        import clinical_knowledge.store as core_store
        assert core_store.MKBStore is MKBStore

    def test_existing_cka_blocks_still_import(self):
        # If SEC-01 broke any existing module, these would fail.
        import clinical_knowledge.preflight    # noqa: F401
        import clinical_knowledge.scaffold     # noqa: F401
        import clinical_knowledge.consensus.engine    # noqa: F401
        import clinical_knowledge.connectors.registry    # noqa: F401
        import app.clinical_knowledge_safety_viewer    # noqa: F401


# ---------------------------------------------------------------------------
# TestPublicReport
# ---------------------------------------------------------------------------

class TestPublicReport:
    @pytest.fixture(scope="class")
    def report_dict(self):
        from scripts.run_cka_sec01_sqlcipher_encryption_validation import run_validation
        return run_validation()

    def test_block_id(self, report_dict):
        assert report_dict["block_id"] == "CKA-SEC-01"

    def test_conclusion_valid(self, report_dict):
        assert report_dict["conclusion"] in (
            "cka_sec01_sqlcipher_encrypted_store_ready",
            "cka_sec01_sqlcipher_provider_required",
        )

    def test_main_store_flags_false(self, report_dict):
        assert report_dict["sqlcipher_encryption_active_for_main_store"] is False
        assert report_dict["main_store_migration_performed"] is False
        assert report_dict["real_data_migrated"] is False

    def test_safety_flags_false(self, report_dict):
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
            assert report_dict[k] is False, f"flag {k!r} not False"

    def test_zero_leak_counters(self, report_dict):
        assert report_dict["private_filename_path_leaks"] == 0
        assert report_dict["secret_leaks"] == 0

    def test_report_contains_no_key(self, report_dict):
        text = json.dumps(report_dict)
        assert "synthkey_" not in text

    def test_report_contains_no_drive_letter_path(self, report_dict):
        text = json.dumps(report_dict)
        assert not re.search(r"[A-Za-z]:\\\\", text)

    def test_report_contains_no_temp_db_path(self, report_dict):
        text = json.dumps(report_dict)
        assert "cka_sec01_synth_" not in text
        assert "cka_sec01_test_" not in text

    def test_report_passes_b02_privacy_checker(self, report_dict):
        from clinical_knowledge.privacy.report_privacy import (
            check_public_report_payload,
        )
        result = check_public_report_payload(report_dict)
        assert result.passed, (
            f"privacy checker rejected report: "
            f"{result.leak_examples_redacted}"
        )

    def test_next_recommended_block_present(self, report_dict):
        nxt = report_dict.get("next_recommended_block", "")
        assert isinstance(nxt, str)
        assert "SEC-02" in nxt

    def test_report_files_written(self):
        d = REPO_ROOT / "reports" / "cka_sec01_sqlcipher_encryption"
        assert (d / "cka_sec01_sqlcipher_encryption_report.json").exists()
        assert (d / "cka_sec01_sqlcipher_encryption_report.md").exists()


# ---------------------------------------------------------------------------
# TestNoClinicalLogicChange
# ---------------------------------------------------------------------------

class TestNoClinicalLogicChange:
    """SEC-01 must not change clinical logic, OCR, extractor, or safety gates."""

    def test_no_clinical_text_in_security_package(self):
        # The security package must not contain medical advice strings.
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


# ---------------------------------------------------------------------------
# TestVerifyHelpersBehavior — provider-agnostic
# ---------------------------------------------------------------------------

class TestVerifyHelpersBehavior:
    def test_verify_cipher_version_handles_none(self):
        assert verify_cipher_version(None) is None

    def test_verify_wrong_key_returns_false_when_provider_unavailable(self):
        if _has_provider():
            pytest.skip("provider available — testing the unavailable path only")
        # The helper returns False when no provider is present so callers
        # mark it skipped, NOT passed.
        assert verify_wrong_key_fails(":memory:", "irrelevant") is False

    def test_verify_plaintext_absent_handles_missing_file(self):
        nonexistent = str(REPO_ROOT / "does_not_exist_cka_sec01.db")
        # Missing file => cannot verify => False.
        assert verify_plaintext_absent(nonexistent) is False

    def test_verify_plaintext_absent_detects_plaintext(self, tmp_path):
        # Write a synthetic plaintext file containing the forbidden marker.
        f = tmp_path / "plain.db"
        f.write_bytes(b"some prefix SYNTHETIC_PRIVATE_NAME_ALPHA suffix")
        assert verify_plaintext_absent(str(f), SYNTHETIC_FORBIDDEN_STRINGS) is False

    def test_verify_plaintext_absent_passes_clean_file(self, tmp_path):
        f = tmp_path / "clean.db"
        f.write_bytes(b"\x00\x01\x02 random bytes here, none of the markers")
        assert verify_plaintext_absent(str(f), SYNTHETIC_FORBIDDEN_STRINGS) is True
