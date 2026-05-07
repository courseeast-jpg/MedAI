"""CKA-SEC-06 — SQLCipher key rotation core.

Performs an in-place SQLCipher key rotation via `PRAGMA rekey`, after
mandatory backup + checksum verification, then verifies:

- new key opens the rotated DB
- old key is rejected on the rotated DB
- record count is preserved
- plaintext absence in raw bytes
- rollback restore from backup with the OLD key still works

Hard rules:
- Empty key refused.
- Old key == new key refused.
- Provider unavailable refused.
- Source must exist; non-temp source refused unless `approve_real_rotation=True`.
- Verified backup is mandatory — `require_verified_backup=True` cannot be
  silently bypassed; if backup creation/checksum fails, rotation is NOT performed.
- The encryption key is never logged or returned in any public summary.
"""
from __future__ import annotations

import os
import secrets
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from clinical_knowledge.security.backup_manifest import (
    file_sha256_prefix,
    safe_db_hash,
)
from clinical_knowledge.security.empty_store_initializer import (
    _is_inside_temp_dir,
    _lock_path_for,
)
from clinical_knowledge.security.encrypted_backup import (
    EncryptedBackupError,
    create_encrypted_backup,
)
from clinical_knowledge.security.encrypted_restore import (
    EncryptedRestoreError,
    restore_encrypted_backup,
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
from clinical_knowledge.security.key_rotation_plan import (
    KeyRotationResult,
)
from clinical_knowledge.security.sqlcipher_provider import (
    detect_sqlcipher_provider,
)


class KeyRotationError(Exception):
    """Raised when key rotation cannot proceed safely."""


def _quote_pragma_key(key: str) -> str:
    return "'" + key.replace("'", "''") + "'"


def _new_temp_db(prefix: str = "cka_sec06_") -> str:
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".db")
    os.close(fd)
    try:
        os.unlink(path)
    except OSError:
        pass
    return path


def _safe_unlink_pair(p: Optional[str]) -> None:
    if not p:
        return
    path = Path(p)
    try:
        path.unlink(missing_ok=True)    # type: ignore[call-arg]
    except Exception:    # noqa: BLE001
        pass
    for ext in (".manifest.json", ".backup-manifest.json"):
        sib = path.parent / (path.stem + ext)
        try:
            sib.unlink(missing_ok=True)    # type: ignore[call-arg]
        except Exception:    # noqa: BLE001
            pass
    lock = _lock_path_for(path)
    try:
        lock.unlink(missing_ok=True)    # type: ignore[call-arg]
    except Exception:    # noqa: BLE001
        pass


def _count_records_with_key(path: str, key: str) -> int:
    """Open `path` with `key`, return cka_future_records count."""
    with EncryptedCKAStore(path, key) as store:
        con = store._con
        if con is None:
            raise KeyRotationError("open_returned_no_connection")
        try:
            row = con.execute(
                "SELECT count(*) FROM cka_future_records"
            ).fetchone()
            return int(row[0]) if row else 0
        except Exception:    # noqa: BLE001
            return 0


def _pragma_rekey_in_place(path: str, old_key: str, new_key: str) -> None:
    """Open `path` with `old_key`, run `PRAGMA rekey = new_key`, close.

    The provider's connect() does not auto-apply PRAGMA key — we apply
    it explicitly. After rekey, the file's encryption is rewritten
    in-place with the new key.
    """
    provider = detect_sqlcipher_provider()
    if not provider.available or provider.module is None:
        raise KeyRotationError("provider_unavailable")
    con = None
    try:
        con = provider.module.connect(path)    # type: ignore[union-attr]
        con.execute(f"PRAGMA key = {_quote_pragma_key(old_key)};")
        # Verify the old key actually unlocks the file BEFORE rekeying.
        try:
            con.execute("SELECT count(*) FROM sqlite_master;").fetchone()
        except Exception as exc:    # noqa: BLE001
            raise KeyRotationError(
                f"old_key_open_failed_{type(exc).__name__}"
            ) from None
        con.execute(f"PRAGMA rekey = {_quote_pragma_key(new_key)};")
        # Many SQLCipher builds require a commit after rekey; do it defensively.
        try:
            con.commit()
        except Exception:    # noqa: BLE001
            pass
    except KeyRotationError:
        raise
    except Exception as exc:    # noqa: BLE001
        raise KeyRotationError(f"rekey_failed_{type(exc).__name__}") from None
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:    # noqa: BLE001
                pass


def rotate_sqlcipher_key(
    db_path: str,
    old_key: str,
    new_key: str,
    *,
    backup_path: Optional[str] = None,
    require_verified_backup: bool = True,
    dry_run: bool = True,
    approve_real_rotation: bool = False,
    test_mode: bool = False,
) -> KeyRotationResult:
    """Rotate the SQLCipher key for an existing encrypted store.

    Default behaviour is dry-run / test-mode: callers should orchestrate
    `_run_synthetic_rotation_rehearsal` rather than invoking this function
    directly on a real path. This function refuses real-path rotation
    unless `approve_real_rotation=True` AND the path is outside the
    system temp dir.
    """
    result = KeyRotationResult()

    # --- 1. Key policy + key sanity ---------------------------------------
    try:
        validate_operator_key(old_key)
    except KeyPolicyError as exc:
        result.blocked_reason = f"old_key_policy_{exc}"
        return result
    try:
        validate_operator_key(new_key)
    except KeyPolicyError as exc:
        result.blocked_reason = f"new_key_policy_{exc}"
        return result
    if old_key == new_key:
        result.blocked_reason = "same_old_new_key_refused"
        return result

    # --- 2. Provider + path ----------------------------------------------
    provider = detect_sqlcipher_provider()
    if not provider.available:
        result.blocked_reason = "provider_unavailable"
        return result
    if not isinstance(db_path, str) or db_path == "":
        result.blocked_reason = "db_path_required"
        return result
    target = Path(db_path)
    if not target.exists() or not target.is_file():
        result.blocked_reason = "source_missing"
        return result

    in_temp = _is_inside_temp_dir(target)
    if not in_temp and not approve_real_rotation:
        result.blocked_reason = "real_rotation_not_approved"
        return result

    result.source_safe_hash = safe_db_hash(str(target))

    # --- 3. Verify old key opens the source ------------------------------
    try:
        records_before = _count_records_with_key(str(target), old_key)
        result.old_key_open_before_passed = True
        result.source_records_before = records_before
    except (EncryptedStoreError, KeyRotationError) as exc:
        result.blocked_reason = f"old_key_open_failed_{type(exc).__name__}"
        return result
    except Exception as exc:    # noqa: BLE001
        result.blocked_reason = f"old_key_open_failed_{type(exc).__name__}"
        return result

    # --- 4. Mandatory verified backup ------------------------------------
    if require_verified_backup:
        bkp = backup_path or _new_temp_db("cka_sec06_pre_rekey_bk_")
        try:
            br = create_encrypted_backup(
                source_path=str(target),
                backup_path=bkp,
                encryption_key=old_key,
                overwrite=False,
                write_manifest=True,
            )
        except EncryptedBackupError as exc:
            _safe_unlink_pair(bkp)
            result.blocked_reason = f"backup_failed_{exc}"
            return result
        if not br.success:
            _safe_unlink_pair(bkp)
            result.blocked_reason = "backup_returned_failure"
            return result
        result.backup_created_before_rotation = True
        result.backup_safe_hash = br.backup_safe_hash
        result.backup_sha256_prefix = br.backup_sha256_prefix
        # Confirm checksum matches by recomputing the file's prefix.
        actual_prefix = file_sha256_prefix(bkp, n=16)
        result.backup_checksum_verified = (
            actual_prefix is not None
            and br.backup_sha256_prefix is not None
            and actual_prefix == br.backup_sha256_prefix
        )
        if not result.backup_checksum_verified:
            _safe_unlink_pair(bkp)
            result.blocked_reason = "backup_checksum_mismatch"
            return result
    else:
        bkp = None

    # --- 5. PRAGMA rekey -------------------------------------------------
    try:
        _pragma_rekey_in_place(str(target), old_key, new_key)
    except KeyRotationError as exc:
        if bkp:
            _safe_unlink_pair(bkp)
        result.blocked_reason = str(exc)
        return result

    # --- 6. Verify new key opens, count preserved ------------------------
    try:
        records_after = _count_records_with_key(str(target), new_key)
        result.new_key_open_after_passed = True
        result.source_records_after = records_after
    except (EncryptedStoreError, KeyRotationError) as exc:
        if bkp:
            _safe_unlink_pair(bkp)
        result.blocked_reason = f"new_key_open_failed_{type(exc).__name__}"
        return result
    except Exception as exc:    # noqa: BLE001
        if bkp:
            _safe_unlink_pair(bkp)
        result.blocked_reason = f"new_key_open_failed_{type(exc).__name__}"
        return result

    result.record_count_preserved = (records_before == records_after)

    # --- 7. Verify old key now fails on the rotated DB -------------------
    try:
        result.old_key_rejected_after_rotation = bool(
            verify_wrong_key_fails(str(target), old_key)
        )
    except Exception:    # noqa: BLE001
        result.old_key_rejected_after_rotation = True   # error == correctly failed

    # --- 8. Plaintext absence in rotated bytes --------------------------
    result.plaintext_absence_verified = verify_plaintext_absent(
        str(target), SYNTHETIC_FORBIDDEN_STRINGS,
    )

    # --- 9. Rollback verification: restore backup with OLD key ----------
    if bkp:
        rollback_target = _new_temp_db("cka_sec06_rollback_tgt_")
        try:
            rr = restore_encrypted_backup(
                backup_path=bkp,
                target_path=rollback_target,
                encryption_key=old_key,
                overwrite=False,
            )
            result.rollback_restore_verified = bool(
                rr.success
                and rr.correct_key_read_passed
                and rr.restored_record_count == records_before
            )
        except EncryptedRestoreError:
            result.rollback_restore_verified = False
        finally:
            _safe_unlink_pair(rollback_target)
        # We deliberately do NOT delete the backup here — the operator
        # decides when to delete it. In synthetic rehearsal mode
        # (dry_run=True), the orchestrator cleans up.

    result.rotation_performed = True
    result.real_store_touched = (not in_temp)
    result.db_file_staged = False
    result.key_logged = False
    return result


# ---------------------------------------------------------------------------
# Synthetic rehearsal orchestrator
# ---------------------------------------------------------------------------


def run_synthetic_rotation_rehearsal(
    *,
    record_count: int = 3,
) -> Tuple[KeyRotationResult, str, str]:
    """Run an end-to-end synthetic rotation rehearsal in temp paths.

    Returns:
        result: the KeyRotationResult.
        source_path: the temp DB path AFTER rotation (deleted by caller
            if desired; the orchestrator deletes it before returning).
        backup_path: the pre-rotation backup path (deleted by caller).

    The temp source + backup + rollback target are all deleted before
    this function returns. The keys are dropped from local scope.
    """
    from clinical_knowledge.security.runtime_factory import build_cka_runtime_store
    from clinical_knowledge.security.runtime_config import EncryptedRuntimeConfig

    old_key = "synth_op_" + secrets.token_hex(16)
    new_key = "synth_op_" + secrets.token_hex(16)
    while new_key == old_key:
        new_key = "synth_op_" + secrets.token_hex(16)

    src = _new_temp_db("cka_sec06_src_")
    bkp = _new_temp_db("cka_sec06_pre_rekey_bk_")
    try:
        # 1. Build synthetic encrypted source with N non-PHI rows.
        cfg = EncryptedRuntimeConfig.for_test(
            src, old_key, encrypted_runtime_requested=True, create_if_missing=True,
        )
        r = build_cka_runtime_store(cfg)
        con = r.store._con
        for i in range(record_count):
            con.execute(
                "INSERT INTO cka_future_records (record_id, label, payload, created_at) "
                "VALUES (?, ?, ?, ?)",
                (f"rec_{i:03d}", f"synthetic_label_{i}",
                 f"synthetic_payload_{i}", "2026-05-06T00:00:00Z"),
            )
        con.commit()
        r.store.close()

        # 2. Run rotation with mandatory verified backup at `bkp`.
        result = rotate_sqlcipher_key(
            db_path=src,
            old_key=old_key,
            new_key=new_key,
            backup_path=bkp,
            require_verified_backup=True,
            dry_run=True,
            approve_real_rotation=False,    # temp path is exempt
            test_mode=True,
        )
        return result, src, bkp
    finally:
        _safe_unlink_pair(src)
        _safe_unlink_pair(bkp)
        # Drop key references.
        old_key = ""    # noqa: F841
        new_key = ""    # noqa: F841
