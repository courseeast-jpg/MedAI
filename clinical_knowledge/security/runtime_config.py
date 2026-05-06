"""CKA-SEC-04 — encrypted runtime configuration.

Reads operator environment flags into a structured config without ever
logging or returning the encryption key.

Environment variables (all OFF by default):
    MEDAI_CKA_ENCRYPTED_STORE_ENABLED         "1"/"true"/"yes" enables
    MEDAI_CKA_ENCRYPTED_STORE_PATH            DB target path
    MEDAI_CKA_ENCRYPTION_KEY                  operator-supplied key
    MEDAI_CKA_ENCRYPTED_STORE_CREATE_IF_MISSING "1"/"true" allows fresh-create

Test-only:
    CKA_SEC04_TEST_KEY                        used only when test_mode=True
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Optional

from clinical_knowledge.security.sqlcipher_provider import (
    detect_sqlcipher_provider,
)


_PATH_HASH_SALT = "cka_sec04_runtime_path_v1"

_ENV_ENABLED = "MEDAI_CKA_ENCRYPTED_STORE_ENABLED"
_ENV_PATH = "MEDAI_CKA_ENCRYPTED_STORE_PATH"
_ENV_KEY = "MEDAI_CKA_ENCRYPTION_KEY"
_ENV_CREATE = "MEDAI_CKA_ENCRYPTED_STORE_CREATE_IF_MISSING"
_ENV_TEST_KEY = "CKA_SEC04_TEST_KEY"


def _truthy(raw: Optional[str]) -> bool:
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _safe_path_hash(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    digest = hashlib.sha256(f"{_PATH_HASH_SALT}:{raw}".encode("utf-8")).hexdigest()[:16]
    return f"cka_db_{digest}"


def _safe_path_label(raw: Optional[str]) -> Optional[str]:
    """Return a non-revealing label for a path: only the base filename, not the dir.

    Even the basename is opt-in via the safe label only — the public report
    primarily uses the safe hash.
    """
    if not raw:
        return None
    base = os.path.basename(raw)
    if not base:
        return None
    # Truncate suspicious-looking long basenames defensively.
    return base[:48]


@dataclass
class EncryptedRuntimeConfig:
    """Public-report-safe configuration derived from environment variables.

    The encryption key is NEVER stored on this dataclass. We only carry a
    boolean `key_present` flag.
    """

    encrypted_runtime_requested: bool = False
    encrypted_store_path_safe_label: Optional[str] = None
    encrypted_store_path_hash: Optional[str] = None
    key_present: bool = False
    provider_available: bool = False
    create_if_missing: bool = False
    test_mode: bool = False

    # Internal-only path; never written to public summary.
    _path: Optional[str] = field(default=None, repr=False)
    # Internal-only key reference (string). Never serialized.
    _key: Optional[str] = field(default=None, repr=False)

    @classmethod
    def from_env(
        cls,
        env: Optional[dict] = None,
        *,
        test_mode: bool = False,
    ) -> "EncryptedRuntimeConfig":
        e = env if env is not None else os.environ
        enabled = _truthy(e.get(_ENV_ENABLED))
        path = e.get(_ENV_PATH) or None
        key = e.get(_ENV_KEY) or None
        create = _truthy(e.get(_ENV_CREATE))
        if test_mode and not key:
            # Test-mode-only fallback. Never touched in non-test mode.
            tk = e.get(_ENV_TEST_KEY)
            if tk:
                key = tk
        provider_status = detect_sqlcipher_provider()
        return cls(
            encrypted_runtime_requested=enabled,
            encrypted_store_path_safe_label=_safe_path_label(path),
            encrypted_store_path_hash=_safe_path_hash(path),
            key_present=bool(key),
            provider_available=bool(provider_status.available),
            create_if_missing=create,
            test_mode=bool(test_mode),
            _path=path,
            _key=key,
        )

    @classmethod
    def for_test(
        cls,
        path: str,
        key: str,
        *,
        encrypted_runtime_requested: bool = True,
        create_if_missing: bool = False,
    ) -> "EncryptedRuntimeConfig":
        provider_status = detect_sqlcipher_provider()
        return cls(
            encrypted_runtime_requested=encrypted_runtime_requested,
            encrypted_store_path_safe_label=_safe_path_label(path),
            encrypted_store_path_hash=_safe_path_hash(path),
            key_present=bool(key),
            provider_available=bool(provider_status.available),
            create_if_missing=create_if_missing,
            test_mode=True,
            _path=path,
            _key=key,
        )

    @property
    def runtime_activation_allowed(self) -> bool:
        """All-of: requested AND provider AND key. Migration is forbidden."""
        return (
            self.encrypted_runtime_requested
            and self.provider_available
            and self.key_present
        )

    # Internal accessors used by factory/preflight only — NEVER include in
    # public summary or stringification.
    def _resolved_path(self) -> Optional[str]:
        return self._path

    def _resolved_key(self) -> Optional[str]:
        return self._key

    def safe_public_summary(self) -> dict:
        return {
            "encrypted_runtime_requested": self.encrypted_runtime_requested,
            "encrypted_store_path_safe_label": self.encrypted_store_path_safe_label,
            "encrypted_store_path_hash": self.encrypted_store_path_hash,
            "key_present": self.key_present,
            "provider_available": self.provider_available,
            "create_if_missing": self.create_if_missing,
            "test_mode": self.test_mode,
            "runtime_activation_allowed": self.runtime_activation_allowed,
        }
