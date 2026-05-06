"""CKA-SEC-07 — encrypted backup manifest.

Public-report-safe descriptor for an encrypted-store backup. The
manifest is written as a sibling JSON next to the backup file. It
NEVER carries the encryption key, NEVER carries raw filesystem paths;
only stable safe hashes and integrity metadata.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


_SAFE_HASH_SALT = "cka_sec07_backup_v1"


def _safe_hash(raw: str) -> str:
    digest = hashlib.sha256(f"{_SAFE_HASH_SALT}:{raw}".encode("utf-8")).hexdigest()[:16]
    return f"cka_db_{digest}"


def safe_db_hash(path: str) -> str:
    """Return a stable, public-report-safe hash for any DB path."""
    return _safe_hash(path)


def file_sha256(path: str) -> Optional[str]:
    """Full hex SHA-256 of `path`'s bytes; None if missing/unreadable."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    try:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def file_sha256_prefix(path: str, n: int = 16) -> Optional[str]:
    """First `n` hex chars of SHA-256 — short enough to bypass the B02
    SECRET regex (`[A-Za-z0-9]{40,}`) when serialized in public reports.
    """
    full = file_sha256(path)
    if full is None:
        return None
    return full[: max(1, min(n, len(full)))]


@dataclass
class BackupManifest:
    """Public-report-safe backup manifest.

    Carries enough metadata to verify a future restore round-trip:
    - safe hashes of source + backup paths (for cross-reference)
    - SHA-256 checksum prefix of the backup file (no full 40-char hex)
    - source record count (zero or low integer)
    - schema/manifest version, timestamps

    Hard invariants enforced via __post_init__:
    - backup_kind must be "synthetic_encrypted_byte_copy"
    - encryption_key_logged must be False
    - real_data_in_backup must be False
    - source_record_count must be a non-negative int
    """

    manifest_id: str
    created_at: str
    source_safe_id: str
    backup_safe_id: str
    backup_sha256_prefix: Optional[str]
    source_record_count: int = 0
    schema_version: int = 1
    backup_kind: str = "synthetic_encrypted_byte_copy"

    # Negative assertions:
    real_data_in_backup: bool = False
    encryption_key_logged: bool = False
    key_stored_in_repo: bool = False

    def __post_init__(self) -> None:
        if self.backup_kind != "synthetic_encrypted_byte_copy":
            raise ValueError("backup_kind_must_be_synthetic_encrypted_byte_copy")
        if self.encryption_key_logged is True:
            raise ValueError("encryption_key_logged_not_permitted")
        if self.key_stored_in_repo is True:
            raise ValueError("key_stored_in_repo_not_permitted")
        if self.real_data_in_backup is True:
            raise ValueError("real_data_in_backup_not_permitted_in_sec07")
        if not isinstance(self.source_record_count, int):
            raise ValueError("source_record_count_must_be_int")
        if self.source_record_count < 0:
            raise ValueError("source_record_count_must_be_nonneg")

    @classmethod
    def for_new_backup(
        cls,
        source_path: str,
        backup_path: str,
        backup_sha256_prefix: Optional[str],
        source_record_count: int,
    ) -> "BackupManifest":
        return cls(
            manifest_id=f"cka_sec07_backup_manifest_{uuid.uuid4().hex[:12]}",
            created_at=datetime.now(timezone.utc).isoformat(),
            source_safe_id=safe_db_hash(source_path) if source_path else "cka_db_none",
            backup_safe_id=safe_db_hash(backup_path) if backup_path else "cka_db_none",
            backup_sha256_prefix=backup_sha256_prefix,
            source_record_count=int(source_record_count),
        )

    def safe_public_summary(self) -> dict:
        return {
            "manifest_id": self.manifest_id,
            "created_at": self.created_at,
            "source_safe_id": self.source_safe_id,
            "backup_safe_id": self.backup_safe_id,
            "backup_sha256_prefix": self.backup_sha256_prefix,
            "source_record_count": self.source_record_count,
            "schema_version": self.schema_version,
            "backup_kind": self.backup_kind,
            "real_data_in_backup": self.real_data_in_backup,
            "encryption_key_logged": self.encryption_key_logged,
            "key_stored_in_repo": self.key_stored_in_repo,
        }


def manifest_path_for(backup_path: str) -> Path:
    """Return the sibling manifest path for a given backup file."""
    p = Path(backup_path)
    return p.parent / (p.stem + ".backup-manifest.json")


def write_backup_manifest(backup_path: str, manifest: BackupManifest) -> str:
    """Write the manifest JSON next to the backup file. Returns its path."""
    target = manifest_path_for(backup_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(manifest.safe_public_summary(), indent=2),
        encoding="utf-8",
    )
    return str(target)


def read_backup_manifest(backup_path: str) -> Optional[dict]:
    """Read the manifest JSON next to the backup file. Returns None if missing."""
    p = manifest_path_for(backup_path)
    if not p.exists() or not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:    # noqa: BLE001
        return None
