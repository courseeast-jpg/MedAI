"""CKA-SEC-04 — runtime factory for the CKA store.

If the operator has explicitly enabled the encrypted runtime via the
SEC-04 environment flags AND a key + provider are available, the
factory returns an `EncryptedCKAStore`. Otherwise it returns the
default unencrypted `MKBStore`.

Hard rules:
- No data migration. The factory NEVER copies records from MKBStore.
- No silent fallback. If the operator REQUESTED the encrypted runtime
  and it cannot be opened, the factory raises — it does NOT silently
  return the unencrypted store.
- No file overwrite. If the encrypted target does not exist and
  `create_if_missing` is False, the factory raises.
- The encryption key is never logged or returned.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from clinical_knowledge.security.encrypted_store import (
    EncryptedCKAStore,
    EncryptedStoreError,
)
from clinical_knowledge.security.empty_store_initializer import (
    EmptyStoreInitError,
    initialize_empty_encrypted_store,
)
from clinical_knowledge.security.runtime_config import EncryptedRuntimeConfig
from clinical_knowledge.security.sqlcipher_provider import (
    detect_sqlcipher_provider,
)
from clinical_knowledge.store import MKBStore


class RuntimeFactoryError(Exception):
    """Raised when an explicit encrypted-runtime request cannot be honoured."""


@dataclass
class RuntimeBuildResult:
    """Public-report-safe outcome of a runtime build call.

    The store object itself is held internally for use by the caller.
    Public summaries expose only flags / classifications.
    """

    runtime_encryption_active: bool = False
    encrypted_store_opened: bool = False
    encrypted_store_created: bool = False
    fallback_to_mkbstore: bool = False
    main_store_migration_performed: bool = False
    real_data_migrated: bool = False
    store_kind: str = "unknown"
    store: Optional[Union[MKBStore, EncryptedCKAStore]] = None

    def safe_public_summary(self) -> dict:
        return {
            "runtime_encryption_active": self.runtime_encryption_active,
            "encrypted_store_opened": self.encrypted_store_opened,
            "encrypted_store_created": self.encrypted_store_created,
            "fallback_to_mkbstore": self.fallback_to_mkbstore,
            "main_store_migration_performed": self.main_store_migration_performed,
            "real_data_migrated": self.real_data_migrated,
            "store_kind": self.store_kind,
        }


def _open_encrypted(
    path: str, key: str,
) -> EncryptedCKAStore:
    """Open an existing encrypted store. Wrong key raises."""
    return EncryptedCKAStore(path, key)


def _create_empty_encrypted(
    path: str, key: str,
) -> None:
    """Create an empty encrypted store at `path`. Refuses to overwrite."""
    p = Path(path)
    if p.exists():
        raise RuntimeFactoryError("encrypted_target_exists_will_not_overwrite")
    # The initializer enforces empty key, hardcoded markers, key policy.
    # We pass approve_real_store_creation=True ONLY because the operator's
    # enabling of the runtime flag is treated as the equivalent operator
    # signal at this layer; the SEC-03A operator guide documents that
    # SEC-03A handles the human approval at file-creation time.
    initialize_empty_encrypted_store(
        target_path=path,
        encryption_key=key,
        approve_real_store_creation=True,
        overwrite=False,
        create_manifest=True,
    )


def build_cka_runtime_store(
    config: EncryptedRuntimeConfig,
    *,
    test_mode: bool = False,
) -> RuntimeBuildResult:
    """Return the appropriate CKA store based on operator configuration.

    Default behavior (no env flags): returns an in-memory MKBStore with
    `runtime_encryption_active=False`.

    With encrypted runtime requested: opens or (if `create_if_missing`)
    creates an empty encrypted store. Wrong key raises. Missing path
    without `create_if_missing` raises. No silent fallback.
    """
    if not config.encrypted_runtime_requested:
        # Default path — unencrypted MKBStore.
        store = MKBStore(":memory:")
        return RuntimeBuildResult(
            runtime_encryption_active=False,
            encrypted_store_opened=False,
            encrypted_store_created=False,
            fallback_to_mkbstore=False,
            main_store_migration_performed=False,
            real_data_migrated=False,
            store_kind="mkbstore_unencrypted_default",
            store=store,
        )

    # Encrypted runtime explicitly requested. From here on, NO silent fallback.
    if not config.provider_available:
        raise RuntimeFactoryError("sqlcipher_provider_unavailable_for_encrypted_runtime")
    if not config.key_present:
        raise RuntimeFactoryError("encrypted_runtime_blocked_missing_key")

    path = config._resolved_path()
    if not path:
        raise RuntimeFactoryError("encrypted_runtime_missing_path")

    key = config._resolved_key()
    if not key:
        # Should not happen given key_present check, but defensive.
        raise RuntimeFactoryError("encrypted_runtime_blocked_missing_key")

    target = Path(path)
    created = False
    if not target.exists():
        if not config.create_if_missing:
            raise RuntimeFactoryError("encrypted_target_missing_create_if_missing_false")
        try:
            _create_empty_encrypted(path, key)
            created = True
        except (EmptyStoreInitError, EncryptedStoreError) as exc:
            raise RuntimeFactoryError(
                f"encrypted_runtime_create_failed_{type(exc).__name__}"
            ) from None

    # Open with the operator-supplied key. Wrong key raises explicitly.
    try:
        store = _open_encrypted(path, key)
    except EncryptedStoreError as exc:
        raise RuntimeFactoryError(
            f"encrypted_runtime_open_failed_{type(exc).__name__}"
        ) from None
    except Exception as exc:    # noqa: BLE001
        # SQLCipher providers may raise sqlite3 errors on wrong-key opens.
        raise RuntimeFactoryError(
            f"encrypted_runtime_open_failed_{type(exc).__name__}"
        ) from None

    return RuntimeBuildResult(
        runtime_encryption_active=True,
        encrypted_store_opened=True,
        encrypted_store_created=created,
        fallback_to_mkbstore=False,
        main_store_migration_performed=False,
        real_data_migrated=False,
        store_kind="encrypted_ckastore",
        store=store,
    )
