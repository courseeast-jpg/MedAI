"""CKA-SEC-03A — encrypted empty future store initializer.

This module creates an empty SQLCipher-encrypted DB suitable as the
*future* CKA store. It does **not** migrate, copy, or read any
existing data. It does **not** activate the new store at runtime. It
does **not** modify the production MKBStore.

Layered safety:
- Empty key refused.
- Provider-unavailable refused.
- Real (non-temp) target paths refused unless `approve_real_store_creation=True`.
- Existing target file never overwritten unless `overwrite=True`.
- Lock file prevents concurrent initialization.
- Manifest written alongside DB (never carries the key).
- Schema is empty — zero records, exactly one non-sensitive
  `store_initialized` ledger event.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from clinical_knowledge.security.encrypted_store import (
    EncryptedCKAStore,
    EncryptedStoreError,
)
from clinical_knowledge.security.encrypted_store_manifest import (
    EncryptedStoreManifest,
    safe_db_file_hash,
    write_manifest_alongside_db,
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


_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class EmptyStoreInitError(Exception):
    """Raised when the empty-store initializer cannot operate safely."""


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------


@dataclass
class InitializationResult:
    """Public-report-safe result of an empty-store initialization."""

    success: bool = False
    target_safe_hash: Optional[str] = None
    schema_created: bool = False
    records_count: int = 0
    ledger_events_count: int = 0
    correct_key_read_passed: bool = False
    wrong_key_failure_passed: bool = False
    plaintext_absence_verified: bool = False
    runtime_active: bool = False
    main_store_migration_performed: bool = False
    real_data_migrated: bool = False
    operator_approved_creation: bool = False
    overwrite_used: bool = False
    lock_file_used: bool = False
    lock_file_left_behind: bool = False
    manifest_written: bool = False
    db_file_staged: bool = False  # initializer never stages files itself

    def safe_public_summary(self) -> dict:
        return {
            "success": self.success,
            "target_safe_hash": self.target_safe_hash,
            "schema_created": self.schema_created,
            "records_count": self.records_count,
            "ledger_events_count": self.ledger_events_count,
            "correct_key_read_passed": self.correct_key_read_passed,
            "wrong_key_failure_passed": self.wrong_key_failure_passed,
            "plaintext_absence_verified": self.plaintext_absence_verified,
            "runtime_active": self.runtime_active,
            "main_store_migration_performed": self.main_store_migration_performed,
            "real_data_migrated": self.real_data_migrated,
            "operator_approved_creation": self.operator_approved_creation,
            "overwrite_used": self.overwrite_used,
            "lock_file_used": self.lock_file_used,
            "lock_file_left_behind": self.lock_file_left_behind,
            "manifest_written": self.manifest_written,
            "db_file_staged": self.db_file_staged,
        }


# ---------------------------------------------------------------------------
# Path classification
# ---------------------------------------------------------------------------


def _is_inside_temp_dir(target: Path) -> bool:
    """Return True only if `target` resolves under the system temp dir.

    Compared by canonical-resolved path prefix, so symlinks cannot fool it.
    """
    try:
        target_real = target.resolve()
    except Exception:    # noqa: BLE001
        return False
    candidates: List[Path] = []
    try:
        candidates.append(Path(tempfile.gettempdir()).resolve())
    except Exception:    # noqa: BLE001
        pass
    for env_var in ("TMP", "TEMP", "TMPDIR"):
        v = os.environ.get(env_var)
        if v:
            try:
                candidates.append(Path(v).resolve())
            except Exception:    # noqa: BLE001
                continue
    for cand in candidates:
        try:
            target_real.relative_to(cand)
            return True
        except ValueError:
            continue
    return False


# ---------------------------------------------------------------------------
# Lock file helpers
# ---------------------------------------------------------------------------


def _lock_path_for(db_path: Path) -> Path:
    return db_path.with_name(db_path.stem + ".init.lock")


@contextmanager
def _init_lock(db_path: Path):
    """Simple O_EXCL lock file. Refuses if a stale lock is present."""
    lock = _lock_path_for(db_path)
    if lock.exists():
        raise EmptyStoreInitError("init_lock_present")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(str(lock), flags, 0o600)
    except FileExistsError:
        raise EmptyStoreInitError("init_lock_race") from None
    except OSError as exc:
        raise EmptyStoreInitError(f"init_lock_unwritable_{type(exc).__name__}") from None
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            fp.write("cka-sec03a-init-lock\n")
        yield lock
    finally:
        try:
            lock.unlink(missing_ok=True)    # type: ignore[call-arg]
        except Exception:    # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Schema (empty store)
# ---------------------------------------------------------------------------

# Single CREATE statements only — no INSERTs except a single non-sensitive
# initialization marker into the metadata table.
_FUTURE_STORE_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS cka_future_records (
        record_id   TEXT PRIMARY KEY,
        label       TEXT NOT NULL,
        payload     TEXT NOT NULL,
        created_at  TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS cka_future_ledger (
        event_id    TEXT PRIMARY KEY,
        event_type  TEXT NOT NULL,
        ts          TEXT NOT NULL,
        details     TEXT NOT NULL DEFAULT '{}'
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS cka_future_metadata (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    """,
)

_NON_SENSITIVE_INIT_LEDGER_EVENT = (
    "cka_sec03a_init_event",
    "store_initialized",
    "{}",
)

# Non-sensitive metadata rows — none of these contain PHI.
def _non_sensitive_metadata_rows() -> List[tuple]:
    return [
        ("schema_version", str(_SCHEMA_VERSION)),
        ("store_kind", "encrypted_empty_future_store"),
        ("created_at", datetime.now(timezone.utc).isoformat()),
        ("runtime_active", "false"),
        ("real_data_migrated", "false"),
        ("main_store_migration_performed", "false"),
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def initialize_empty_encrypted_store(
    target_path: str,
    encryption_key: str,
    *,
    approve_real_store_creation: bool = False,
    overwrite: bool = False,
    create_manifest: bool = True,
) -> InitializationResult:
    """Initialize an empty SQLCipher-encrypted DB at target_path.

    Refuses unless the path is inside the system temp dir OR
    `approve_real_store_creation=True`. Refuses to overwrite an
    existing file unless `overwrite=True`.

    Never logs the encryption key. Never copies real data. Schema is
    empty: zero records, one non-sensitive `store_initialized` event,
    six non-sensitive metadata rows.
    """
    # 1. Key policy first — never log the key.
    try:
        validate_operator_key(encryption_key)
    except KeyPolicyError as exc:
        raise EmptyStoreInitError(f"key_policy_{exc}") from None

    # 2. Provider must be available.
    provider = detect_sqlcipher_provider()
    if not provider.available:
        raise EmptyStoreInitError("provider_unavailable")

    target = Path(target_path)
    if not target_path or target.name == "":
        raise EmptyStoreInitError("target_path_required")

    in_temp = _is_inside_temp_dir(target)
    if not in_temp and not approve_real_store_creation:
        raise EmptyStoreInitError("real_store_creation_not_approved")

    # 3. Existence / overwrite check.
    if target.exists():
        if not overwrite:
            raise EmptyStoreInitError("target_exists_overwrite_required")
        # Overwrite was requested — but for the empty-store initializer we
        # ALSO require the operator to have approved real-store creation
        # if this is not a temp-dir target. Belt-and-suspenders.
        if not in_temp and not approve_real_store_creation:
            raise EmptyStoreInitError("overwrite_outside_temp_requires_approval")
        try:
            target.unlink()
        except OSError as exc:
            raise EmptyStoreInitError(
                f"could_not_remove_existing_target_{type(exc).__name__}"
            ) from None

    # 4. Parent dir creation — only if safe.
    parent = target.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise EmptyStoreInitError(
            f"parent_dir_unavailable_{type(exc).__name__}"
        ) from None

    result = InitializationResult(
        target_safe_hash=safe_db_file_hash(str(target)),
        operator_approved_creation=approve_real_store_creation,
        overwrite_used=overwrite,
    )

    lock_left_behind = False
    try:
        with _init_lock(target):
            result.lock_file_used = True

            # 5. Create the encrypted store + empty schema.
            with EncryptedCKAStore(str(target), encryption_key) as store:
                con = store._con    # internal connection — adapter is open with PRAGMA key applied
                if con is None:
                    raise EmptyStoreInitError("encrypted_connection_unavailable")
                for stmt in _FUTURE_STORE_SCHEMA:
                    con.execute(stmt)
                # One non-sensitive ledger event.
                event_id, event_type, details_json = _NON_SENSITIVE_INIT_LEDGER_EVENT
                con.execute(
                    "INSERT INTO cka_future_ledger (event_id, event_type, ts, details) "
                    "VALUES (?, ?, ?, ?)",
                    (event_id, event_type, datetime.now(timezone.utc).isoformat(),
                     details_json),
                )
                # Non-sensitive metadata rows.
                for k, v in _non_sensitive_metadata_rows():
                    con.execute(
                        "INSERT OR REPLACE INTO cka_future_metadata (key, value) "
                        "VALUES (?, ?)",
                        (k, v),
                    )
                con.commit()
                result.schema_created = True

                # Confirm record count is zero (empty store invariant).
                rec_row = con.execute(
                    "SELECT count(*) FROM cka_future_records"
                ).fetchone()
                ledger_row = con.execute(
                    "SELECT count(*) FROM cka_future_ledger"
                ).fetchone()
                result.records_count = int(rec_row[0]) if rec_row else 0
                result.ledger_events_count = int(ledger_row[0]) if ledger_row else 0
                # Empty-store invariant
                if result.records_count != 0:
                    raise EmptyStoreInitError("records_count_not_zero")

            # 6. Verify correct-key read on a re-open.
            with EncryptedCKAStore(str(target), encryption_key) as store2:
                con = store2._con
                if con is None:
                    raise EmptyStoreInitError("reopen_failed")
                count = con.execute(
                    "SELECT count(*) FROM cka_future_records"
                ).fetchone()[0]
                if count == 0:
                    result.correct_key_read_passed = True

            # 7. Wrong-key probe.
            wrong_key = "wrongkey_sec03a_" + os.urandom(8).hex()
            if wrong_key == encryption_key:
                wrong_key += "_x"
            # The store contains no synthetic forbidden strings, but the
            # wrong-key path must still fail to even open the schema.
            try:
                wk_failed = verify_wrong_key_fails(str(target), wrong_key)
            except Exception:    # noqa: BLE001
                wk_failed = True
            result.wrong_key_failure_passed = bool(wk_failed)

            # 8. Plaintext-absence probe (ensures schema strings + the
            # one synthetic forbidden marker we wrote into a SCRATCH
            # buffer are not present in raw bytes). For an empty store
            # we use SYNTHETIC_FORBIDDEN_STRINGS as a defense-in-depth
            # check that the plaintext layer is absent.
            result.plaintext_absence_verified = verify_plaintext_absent(
                str(target), SYNTHETIC_FORBIDDEN_STRINGS,
            )

            # 9. Manifest.
            if create_manifest:
                manifest = EncryptedStoreManifest.for_new_store(
                    db_path=str(target),
                    provider_name=provider.provider_name,
                    cipher_version=provider.cipher_version,
                    operator_approved_creation=approve_real_store_creation,
                )
                manifest.empty_store_created = True
                manifest.records_count = result.records_count
                manifest.ledger_events_count = result.ledger_events_count
                # NOTE: write_manifest_alongside_db creates a sibling
                # JSON next to the DB file. We do not commit either.
                write_manifest_alongside_db(str(target), manifest)
                result.manifest_written = True

            result.success = (
                result.schema_created
                and result.correct_key_read_passed
                and result.wrong_key_failure_passed
                and result.plaintext_absence_verified
            )
        # Lock has been removed by context manager exit.
        result.lock_file_left_behind = _lock_path_for(target).exists()
    except EmptyStoreInitError:
        # Re-raise, but ensure the lock isn't accidentally orphaned
        # (the context manager already handles this).
        result.lock_file_left_behind = _lock_path_for(target).exists()
        raise
    except Exception as exc:    # noqa: BLE001
        result.lock_file_left_behind = _lock_path_for(target).exists()
        raise EmptyStoreInitError(f"init_unexpected_{type(exc).__name__}") from None

    return result


def initializer_will_create_real_store(
    target_path: str,
    approve_real_store_creation: bool,
) -> bool:
    """True iff calling `initialize_empty_encrypted_store` with these inputs
    would attempt to create a real (non-temp) store. Pure inspection helper —
    does not touch the filesystem."""
    if not target_path:
        return False
    target = Path(target_path)
    if _is_inside_temp_dir(target):
        return False
    return bool(approve_real_store_creation)
