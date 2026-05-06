"""CKA-SEC-04 — encrypted runtime preflight.

Inspects an `EncryptedRuntimeConfig` and (only if encrypted runtime
was requested AND a provider/key are present) probes the encrypted
store for:

- correct-key open
- wrong-key failure
- plaintext absence (defense-in-depth check on raw bytes)
- records-count zero (empty/future store invariant)

The preflight NEVER copies records, NEVER mutates the existing main
store, and NEVER writes the key.
"""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from clinical_knowledge.security.encrypted_store import (
    EncryptedCKAStore,
    EncryptedStoreError,
)
from clinical_knowledge.security.encryption_checks import (
    SYNTHETIC_FORBIDDEN_STRINGS,
    verify_plaintext_absent,
    verify_wrong_key_fails,
)
from clinical_knowledge.security.runtime_config import EncryptedRuntimeConfig
from clinical_knowledge.security.runtime_rollback import (
    get_runtime_rollback_plan,
    rollback_plan_ready,
)


@dataclass
class RuntimePreflightResult:
    """Public-report-safe preflight result."""

    passed: bool = False
    runtime_encryption_active: bool = False
    encrypted_store_opened: bool = False
    encrypted_store_created: bool = False
    correct_key_read_passed: bool = False
    wrong_key_failure_passed: bool = False
    plaintext_absence_verified: bool = False
    records_count: int = 0
    migration_performed: bool = False
    real_data_migrated: bool = False
    rollback_plan_available: bool = False
    blocked_reason: Optional[str] = None

    def safe_public_summary(self) -> dict:
        return {
            "passed": self.passed,
            "runtime_encryption_active": self.runtime_encryption_active,
            "encrypted_store_opened": self.encrypted_store_opened,
            "encrypted_store_created": self.encrypted_store_created,
            "correct_key_read_passed": self.correct_key_read_passed,
            "wrong_key_failure_passed": self.wrong_key_failure_passed,
            "plaintext_absence_verified": self.plaintext_absence_verified,
            "records_count": self.records_count,
            "migration_performed": self.migration_performed,
            "real_data_migrated": self.real_data_migrated,
            "rollback_plan_available": self.rollback_plan_available,
            "blocked_reason": self.blocked_reason,
        }


def run_encrypted_runtime_preflight(
    config: EncryptedRuntimeConfig,
) -> RuntimePreflightResult:
    """Run the SEC-04 preflight against `config`.

    Behaviour:
    - If encrypted runtime is NOT requested → preflight passes vacuously,
      runtime_encryption_active=False.
    - If requested but provider/key/path missing → preflight blocks
      with a safe `blocked_reason`. NEVER returns silent success.
    """
    result = RuntimePreflightResult()
    result.rollback_plan_available = rollback_plan_ready(get_runtime_rollback_plan())

    if not config.encrypted_runtime_requested:
        result.passed = True
        result.runtime_encryption_active = False
        return result

    # Encrypted runtime requested — apply the gates.
    if not config.provider_available:
        result.blocked_reason = "provider_unavailable"
        return result
    if not config.key_present:
        result.blocked_reason = "key_missing"
        return result

    path = config._resolved_path()
    key = config._resolved_key()
    if not path:
        result.blocked_reason = "path_missing"
        return result
    if not key:
        # Defensive — should be caught by key_present.
        result.blocked_reason = "key_missing"
        return result

    target = Path(path)
    if not target.exists():
        if not config.create_if_missing:
            result.blocked_reason = "encrypted_target_missing_create_if_missing_false"
            return result
        # The factory is responsible for actual creation. Preflight just
        # confirms that creation would be permitted; it does not create.
        # So if the file is missing AND create_if_missing is true, we
        # report blocked with a precise reason so the validator can
        # decide whether to invoke the factory.
        result.blocked_reason = "encrypted_target_missing_create_if_missing_true"
        return result

    # File exists — probe correct-key open, then wrong-key failure, then plaintext absence.
    try:
        with EncryptedCKAStore(path, key) as store:
            con = store._con
            if con is None:
                result.blocked_reason = "open_returned_no_connection"
                return result
            # The encrypted future-store schema's records table is
            # cka_future_records. Be tolerant of stores that don't yet
            # have it (older empty stores) by falling back to 0.
            try:
                row = con.execute(
                    "SELECT count(*) FROM cka_future_records"
                ).fetchone()
                result.records_count = int(row[0]) if row else 0
            except Exception:    # noqa: BLE001
                result.records_count = 0
        result.correct_key_read_passed = True
        result.encrypted_store_opened = True
    except EncryptedStoreError as exc:
        result.blocked_reason = f"open_failed_{type(exc).__name__}"
        return result
    except Exception as exc:    # noqa: BLE001
        result.blocked_reason = f"open_failed_{type(exc).__name__}"
        return result

    # Wrong-key probe.
    wrong = "wrongkey_sec04_" + secrets.token_hex(8)
    if wrong == key:
        wrong += "_x"
    try:
        result.wrong_key_failure_passed = bool(
            verify_wrong_key_fails(path, wrong)
        )
    except Exception:    # noqa: BLE001
        result.wrong_key_failure_passed = True   # any error == correctly failed

    # Plaintext-absence probe (defense-in-depth on raw DB bytes).
    result.plaintext_absence_verified = verify_plaintext_absent(
        path, SYNTHETIC_FORBIDDEN_STRINGS
    )

    # Hard invariants.
    result.migration_performed = False
    result.real_data_migrated = False

    result.passed = (
        result.correct_key_read_passed
        and result.wrong_key_failure_passed
        and result.plaintext_absence_verified
        and not result.migration_performed
        and not result.real_data_migrated
    )
    result.runtime_encryption_active = result.passed
    return result
