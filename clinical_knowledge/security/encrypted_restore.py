"""CKA-SEC-07 — encrypted restore helper.

Restores a byte-level encrypted backup (produced by encrypted_backup)
to a target path, verifies the SHA-256 against the manifest, and
verifies that the supplied key opens the restored DB and that the
record count matches the manifest.

Safety invariants:
- empty key refused
- non-existent backup refused
- existing target NOT overwritten unless overwrite=True
- wrong key on the restored DB raises (no silent success)
- encryption key never logged or returned
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from clinical_knowledge.security.backup_manifest import (
    file_sha256_prefix,
    read_backup_manifest,
    safe_db_hash,
)
from clinical_knowledge.security.encrypted_store import (
    EncryptedCKAStore,
    EncryptedStoreError,
)
from clinical_knowledge.security.encryption_checks import (
    SYNTHETIC_FORBIDDEN_STRINGS,
    verify_plaintext_absent,
    verify_wrong_key_fails,
)
from clinical_knowledge.security.key_policy import (
    KeyPolicyError,
    validate_operator_key,
)
from clinical_knowledge.security.sqlcipher_provider import (
    detect_sqlcipher_provider,
)


class EncryptedRestoreError(Exception):
    """Raised when the restore helper cannot operate safely."""


@dataclass
class RestoreResult:
    """Public-report-safe outcome of an encrypted restore."""

    success: bool = False
    backup_safe_hash: Optional[str] = None
    target_safe_hash: Optional[str] = None
    expected_sha256_prefix: Optional[str] = None
    actual_sha256_prefix: Optional[str] = None
    sha256_match: bool = False
    correct_key_read_passed: bool = False
    expected_record_count: int = 0
    restored_record_count: int = 0
    record_count_match: bool = False
    plaintext_absence_verified: bool = False
    overwrite_used: bool = False
    real_data_touched: bool = False
    encryption_key_logged: bool = False
    db_file_staged: bool = False
    blocked_reason: Optional[str] = None

    def safe_public_summary(self) -> dict:
        return {
            "success": self.success,
            "backup_safe_hash": self.backup_safe_hash,
            "target_safe_hash": self.target_safe_hash,
            "expected_sha256_prefix": self.expected_sha256_prefix,
            "actual_sha256_prefix": self.actual_sha256_prefix,
            "sha256_match": self.sha256_match,
            "correct_key_read_passed": self.correct_key_read_passed,
            "expected_record_count": self.expected_record_count,
            "restored_record_count": self.restored_record_count,
            "record_count_match": self.record_count_match,
            "plaintext_absence_verified": self.plaintext_absence_verified,
            "overwrite_used": self.overwrite_used,
            "real_data_touched": self.real_data_touched,
            "encryption_key_logged": self.encryption_key_logged,
            "db_file_staged": self.db_file_staged,
            "blocked_reason": self.blocked_reason,
        }


def _count_records_with_key(target_path: str, key: str) -> int:
    """Open `target_path` with `key` and return cka_future_records count."""
    with EncryptedCKAStore(target_path, key) as store:
        con = store._con
        if con is None:
            raise EncryptedRestoreError("restore_open_failed_no_connection")
        try:
            row = con.execute("SELECT count(*) FROM cka_future_records").fetchone()
            return int(row[0]) if row else 0
        except Exception:    # noqa: BLE001
            return 0


def restore_encrypted_backup(
    backup_path: str,
    target_path: str,
    encryption_key: str,
    *,
    overwrite: bool = False,
) -> RestoreResult:
    """Restore an encrypted backup to `target_path` and verify.

    Steps:
    1. Refuse empty/short/hardcoded keys.
    2. Refuse missing backup or missing manifest sibling.
    3. Refuse existing target unless overwrite=True.
    4. Byte-copy backup to target.
    5. Compute SHA-256 prefix and compare with manifest.
    6. Open target with operator key — wrong key raises.
    7. Confirm record count matches manifest.
    """
    try:
        validate_operator_key(encryption_key)
    except KeyPolicyError as exc:
        raise EncryptedRestoreError(f"key_policy_{exc}") from None

    if not isinstance(backup_path, str) or backup_path == "":
        raise EncryptedRestoreError("backup_path_required")
    if not isinstance(target_path, str) or target_path == "":
        raise EncryptedRestoreError("target_path_required")

    provider = detect_sqlcipher_provider()
    if not provider.available:
        raise EncryptedRestoreError("provider_unavailable")

    backup = Path(backup_path)
    target = Path(target_path)

    if not backup.exists() or not backup.is_file():
        raise EncryptedRestoreError("backup_missing")
    if target.exists() and not overwrite:
        raise EncryptedRestoreError("target_exists_overwrite_required")

    manifest = read_backup_manifest(str(backup))
    if manifest is None:
        raise EncryptedRestoreError("manifest_missing")
    expected_prefix = manifest.get("backup_sha256_prefix")
    expected_count = int(manifest.get("source_record_count", 0))

    # Byte-copy.
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and overwrite:
        try:
            target.unlink()
        except OSError as exc:
            raise EncryptedRestoreError(
                f"target_could_not_remove_existing_{type(exc).__name__}"
            ) from None
    try:
        shutil.copyfile(str(backup), str(target))
    except OSError as exc:
        raise EncryptedRestoreError(
            f"restore_copy_failed_{type(exc).__name__}"
        ) from None

    actual_prefix = file_sha256_prefix(str(target), n=16)
    sha256_match = (
        expected_prefix is not None
        and actual_prefix is not None
        and actual_prefix == expected_prefix
    )

    # Open target with operator key. Wrong key raises EncryptedStoreError
    # which we surface as EncryptedRestoreError.
    try:
        restored_count = _count_records_with_key(str(target), encryption_key)
        correct_key_read = True
    except EncryptedStoreError as exc:
        raise EncryptedRestoreError(
            f"restore_open_failed_{type(exc).__name__}"
        ) from None
    except Exception as exc:    # noqa: BLE001
        raise EncryptedRestoreError(
            f"restore_open_failed_{type(exc).__name__}"
        ) from None

    plaintext_absent = verify_plaintext_absent(str(target), SYNTHETIC_FORBIDDEN_STRINGS)

    return RestoreResult(
        success=(sha256_match and correct_key_read and restored_count == expected_count),
        backup_safe_hash=safe_db_hash(str(backup)),
        target_safe_hash=safe_db_hash(str(target)),
        expected_sha256_prefix=expected_prefix,
        actual_sha256_prefix=actual_prefix,
        sha256_match=sha256_match,
        correct_key_read_passed=correct_key_read,
        expected_record_count=expected_count,
        restored_record_count=restored_count,
        record_count_match=(restored_count == expected_count),
        plaintext_absence_verified=plaintext_absent,
        overwrite_used=overwrite,
        real_data_touched=False,
        encryption_key_logged=False,
        db_file_staged=False,
    )


def verify_restored_wrong_key_fails(target_path: str, wrong_key: str) -> bool:
    """Probe the restored target with a wrong key. Returns True if it fails."""
    return verify_wrong_key_fails(target_path, wrong_key)
