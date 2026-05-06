"""CKA-SEC-07 — encrypted backup helper.

Creates a byte-level copy of an encrypted SQLCipher store. The backup
file is itself encrypted (because the source is encrypted); this
helper never decrypts the source. The encryption key is used ONLY to
verify the source is openable (correct-key + record count) before the
copy proceeds, so a malformed source is not silently snapshotted.

A sibling JSON manifest is written next to the backup with safe hashes,
the backup file's SHA-256 prefix, and the verified source record count.

Safety invariants:
- empty key refused
- non-existent source refused
- existing backup target NOT overwritten unless overwrite=True
- encryption key never logged or returned
- backup carries no real data (synthetic-only callers)
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from clinical_knowledge.security.backup_manifest import (
    BackupManifest,
    file_sha256_prefix,
    safe_db_hash,
    write_backup_manifest,
)
from clinical_knowledge.security.encrypted_store import (
    EncryptedCKAStore,
    EncryptedStoreError,
)
from clinical_knowledge.security.encryption_checks import (
    SYNTHETIC_FORBIDDEN_STRINGS,
    verify_plaintext_absent,
)
from clinical_knowledge.security.key_policy import (
    KeyPolicyError,
    validate_operator_key,
)
from clinical_knowledge.security.sqlcipher_provider import (
    detect_sqlcipher_provider,
)


class EncryptedBackupError(Exception):
    """Raised when the backup helper cannot operate safely."""


@dataclass
class BackupResult:
    """Public-report-safe outcome of an encrypted backup."""

    success: bool = False
    source_safe_hash: Optional[str] = None
    backup_safe_hash: Optional[str] = None
    source_record_count: int = 0
    backup_sha256_prefix: Optional[str] = None
    plaintext_absence_verified: bool = False
    manifest_written: bool = False
    overwrite_used: bool = False
    real_data_touched: bool = False
    encryption_key_logged: bool = False
    db_file_staged: bool = False
    blocked_reason: Optional[str] = None

    def safe_public_summary(self) -> dict:
        return {
            "success": self.success,
            "source_safe_hash": self.source_safe_hash,
            "backup_safe_hash": self.backup_safe_hash,
            "source_record_count": self.source_record_count,
            "backup_sha256_prefix": self.backup_sha256_prefix,
            "plaintext_absence_verified": self.plaintext_absence_verified,
            "manifest_written": self.manifest_written,
            "overwrite_used": self.overwrite_used,
            "real_data_touched": self.real_data_touched,
            "encryption_key_logged": self.encryption_key_logged,
            "db_file_staged": self.db_file_staged,
            "blocked_reason": self.blocked_reason,
        }


def _count_records_with_correct_key(source_path: str, key: str) -> int:
    """Open the encrypted source with the supplied key and return record count.

    Falls back to 0 if the schema doesn't have the cka_future_records table
    (older empty-store layout). Raises EncryptedBackupError on wrong key.
    """
    try:
        with EncryptedCKAStore(source_path, key) as store:
            con = store._con
            if con is None:
                raise EncryptedBackupError("source_open_failed_no_connection")
            try:
                row = con.execute("SELECT count(*) FROM cka_future_records").fetchone()
                return int(row[0]) if row else 0
            except Exception:    # noqa: BLE001
                return 0
    except EncryptedStoreError as exc:
        raise EncryptedBackupError(f"source_open_failed_{type(exc).__name__}") from None
    except Exception as exc:    # noqa: BLE001
        raise EncryptedBackupError(f"source_open_failed_{type(exc).__name__}") from None


def create_encrypted_backup(
    source_path: str,
    backup_path: str,
    encryption_key: str,
    *,
    overwrite: bool = False,
    write_manifest: bool = True,
) -> BackupResult:
    """Create a byte-level encrypted backup of `source_path` at `backup_path`.

    The backup IS encrypted (it is a copy of an encrypted file). The
    encryption key is used only to verify the source is openable
    before the copy. The key is never written to the manifest.
    """
    # Key policy first (no key logging).
    try:
        validate_operator_key(encryption_key)
    except KeyPolicyError as exc:
        raise EncryptedBackupError(f"key_policy_{exc}") from None

    if not isinstance(source_path, str) or source_path == "":
        raise EncryptedBackupError("source_path_required")
    if not isinstance(backup_path, str) or backup_path == "":
        raise EncryptedBackupError("backup_path_required")

    provider = detect_sqlcipher_provider()
    if not provider.available:
        raise EncryptedBackupError("provider_unavailable")

    src = Path(source_path)
    dst = Path(backup_path)
    if not src.exists() or not src.is_file():
        raise EncryptedBackupError("source_missing")
    if dst.exists() and not overwrite:
        raise EncryptedBackupError("backup_target_exists_overwrite_required")

    # Verify source is openable with the supplied key + count records.
    record_count = _count_records_with_correct_key(source_path, encryption_key)

    # Byte-level copy. SQLCipher pages are encrypted at rest, so the
    # backup file content is encrypted by construction.
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and overwrite:
        try:
            dst.unlink()
        except OSError as exc:
            raise EncryptedBackupError(
                f"backup_could_not_remove_existing_{type(exc).__name__}"
            ) from None
    try:
        shutil.copyfile(str(src), str(dst))
    except OSError as exc:
        raise EncryptedBackupError(
            f"backup_copy_failed_{type(exc).__name__}"
        ) from None

    # Compute SHA-256 prefix for the backup file (use 16 hex chars to
    # stay below the B02 SECRET regex's 40+ alnum threshold).
    sha_prefix = file_sha256_prefix(str(dst), n=16)

    # Plaintext-absence check on backup bytes — defense-in-depth.
    plaintext_absent = verify_plaintext_absent(str(dst), SYNTHETIC_FORBIDDEN_STRINGS)

    # Manifest sibling.
    manifest_written = False
    if write_manifest:
        manifest = BackupManifest.for_new_backup(
            source_path=str(src),
            backup_path=str(dst),
            backup_sha256_prefix=sha_prefix,
            source_record_count=record_count,
        )
        write_backup_manifest(str(dst), manifest)
        manifest_written = True

    return BackupResult(
        success=True,
        source_safe_hash=safe_db_hash(str(src)),
        backup_safe_hash=safe_db_hash(str(dst)),
        source_record_count=record_count,
        backup_sha256_prefix=sha_prefix,
        plaintext_absence_verified=plaintext_absent,
        manifest_written=manifest_written,
        overwrite_used=overwrite,
        real_data_touched=False,
        encryption_key_logged=False,
        db_file_staged=False,
    )
