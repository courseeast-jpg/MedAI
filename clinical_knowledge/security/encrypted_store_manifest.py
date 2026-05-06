"""CKA-SEC-03A — encrypted store manifest.

Public-report-safe descriptor for an encrypted empty future CKA store.
- Never carries the encryption key.
- Never carries the raw DB path.
- Carries a 16-hex SHA-256 hash of the path so reports can be cross-referenced
  without leaking the path itself.
"""
from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


_SAFE_HASH_SALT = "cka_sec03a_manifest_v1"


def _safe_hash(raw: str) -> str:
    digest = hashlib.sha256(f"{_SAFE_HASH_SALT}:{raw}".encode("utf-8")).hexdigest()[:16]
    return f"cka_db_{digest}"


def safe_db_file_hash(db_path: str) -> str:
    """Return a stable, public-report-safe hash for any DB path.

    Two calls with the same path return the same hash; calls with different
    paths return different hashes. The raw path is never reachable from the
    hash.
    """
    return _safe_hash(db_path)


@dataclass
class EncryptedStoreManifest:
    """Manifest describing an encrypted empty future store.

    All fields are public-report safe by construction. The class never
    accepts or stores an encryption key.
    """

    manifest_id: str
    created_at: str
    store_safe_id: str
    store_kind: str = "encrypted_empty_future_store"
    db_file_safe_hash: Optional[str] = None
    schema_version: int = 1
    provider_name: Optional[str] = None
    cipher_version: Optional[str] = None

    # Counts — by construction zero on initialization of an empty store
    empty_store_created: bool = False
    records_count: int = 0
    ledger_events_count: int = 0

    # Hard invariants for SEC-03A:
    runtime_active: bool = False
    main_store_migration_performed: bool = False
    real_data_migrated: bool = False

    # Operator gating:
    operator_approved_creation: bool = False

    # Negative assertions:
    key_stored_in_repo: bool = False
    encryption_key_logged: bool = False

    def __post_init__(self) -> None:
        if self.runtime_active is True:
            raise ValueError("runtime_active_must_be_false_in_sec03a")
        if self.main_store_migration_performed is True:
            raise ValueError("main_store_migration_performed_must_be_false_in_sec03a")
        if self.real_data_migrated is True:
            raise ValueError("real_data_migrated_must_be_false_in_sec03a")
        if self.records_count != 0:
            raise ValueError("records_count_must_be_zero_for_empty_store")
        if self.store_kind != "encrypted_empty_future_store":
            raise ValueError("store_kind_must_be_encrypted_empty_future_store")
        if self.key_stored_in_repo is True:
            raise ValueError("key_stored_in_repo_not_permitted")
        if self.encryption_key_logged is True:
            raise ValueError("encryption_key_logged_not_permitted")

    @classmethod
    def for_new_store(
        cls,
        db_path: str,
        provider_name: Optional[str],
        cipher_version: Optional[str],
        operator_approved_creation: bool,
    ) -> "EncryptedStoreManifest":
        return cls(
            manifest_id=f"cka_sec03a_manifest_{uuid.uuid4().hex[:12]}",
            created_at=datetime.now(timezone.utc).isoformat(),
            store_safe_id=safe_db_file_hash(db_path) if db_path else "cka_db_none",
            db_file_safe_hash=safe_db_file_hash(db_path) if db_path else None,
            provider_name=provider_name,
            cipher_version=cipher_version,
            operator_approved_creation=bool(operator_approved_creation),
        )

    def safe_public_summary(self) -> dict:
        """Public-report-safe representation. Never contains the key or path."""
        return {
            "manifest_id": self.manifest_id,
            "created_at": self.created_at,
            "store_safe_id": self.store_safe_id,
            "store_kind": self.store_kind,
            "db_file_safe_hash": self.db_file_safe_hash,
            "schema_version": self.schema_version,
            "provider_name": self.provider_name,
            "cipher_version": self.cipher_version,
            "empty_store_created": self.empty_store_created,
            "records_count": self.records_count,
            "ledger_events_count": self.ledger_events_count,
            "runtime_active": self.runtime_active,
            "main_store_migration_performed": self.main_store_migration_performed,
            "real_data_migrated": self.real_data_migrated,
            "operator_approved_creation": self.operator_approved_creation,
            "key_stored_in_repo": self.key_stored_in_repo,
            "encryption_key_logged": self.encryption_key_logged,
        }


def write_manifest_alongside_db(
    db_path: str,
    manifest: EncryptedStoreManifest,
) -> str:
    """Write the safe manifest as JSON next to the DB file.

    The DB path is never written into the manifest body — only the
    `db_file_safe_hash`. Returns the path of the written manifest.
    """
    import json
    target = Path(db_path)
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    manifest_path = parent / (target.stem + ".manifest.json")
    payload = manifest.safe_public_summary()
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(manifest_path)
