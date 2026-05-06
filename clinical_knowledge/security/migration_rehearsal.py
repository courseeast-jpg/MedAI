"""CKA-SEC-02 — synthetic migration rehearsal.

Creates a *synthetic* unencrypted SQLite source DB in a temp directory,
and rehearses a copy into a SQLCipher-encrypted target DB. All data is
synthetic. The main CKA store is NEVER touched.

Safety invariants:
- Operator-supplied (synthetic for tests) key must satisfy key_policy.
- Source DB is read with stdlib sqlite3 only (no provider needed).
- Target DB is created with the SQLCipher provider via EncryptedCKAStore.
- After rehearsal, both temp files are deleted; their paths are never
  written to public reports.
"""
from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

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
from clinical_knowledge.security.migration_plan import MigrationRehearsalResult
from clinical_knowledge.security.sqlcipher_provider import (
    detect_sqlcipher_provider,
)


# Synthetic records used by the rehearsal. NOT real PHI / NOT real medical data.
SYNTHETIC_RECORDS: tuple = (
    ("rec_001", "label_alpha",
     "SYNTHETIC_PRIVATE_NAME_ALPHA payload SYNTHETIC_MRN_0001"),
    ("rec_002", "label_beta",
     "SYNTHETIC_MEDICAL_NOTE_ALPHA second payload"),
    ("rec_003", "label_gamma",
     "third synthetic record without forbidden marker"),
)

_SOURCE_SCHEMA = """
CREATE TABLE IF NOT EXISTS sec02_synth_source (
    record_id   TEXT PRIMARY KEY,
    label       TEXT NOT NULL,
    payload     TEXT NOT NULL
)
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_sha256(p: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _make_temp_path(prefix: str) -> str:
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".db")
    os.close(fd)
    # Remove the empty file so the provider creates it fresh.
    try:
        os.unlink(path)
    except OSError:
        pass
    return path


def _safe_unlink(path: Optional[str]) -> None:
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)    # type: ignore[call-arg]
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Public rehearsal entrypoint
# ---------------------------------------------------------------------------

def rehearse_synthetic_migration(
    operator_key: str,
    records: Iterable[Tuple[str, str, str]] = SYNTHETIC_RECORDS,
) -> MigrationRehearsalResult:
    """Execute the synthetic migration rehearsal end-to-end.

    Returns a MigrationRehearsalResult with safe public flags.
    The temp files are deleted before this function returns. Their
    paths are never returned, logged, or committed.

    Raises KeyPolicyError if the operator-supplied key violates policy.
    Raises EncryptedStoreError if the SQLCipher provider is unavailable
    (caller should mark rehearsal as not-performed).
    """
    # 1. Enforce key policy BEFORE touching any disk path.
    validate_operator_key(operator_key)

    # 2. Confirm the SQLCipher provider is available.
    provider = detect_sqlcipher_provider()
    if not provider.available:
        raise EncryptedStoreError("provider_unavailable")

    result = MigrationRehearsalResult()
    source_path: Optional[str] = None
    target_path: Optional[str] = None
    try:
        # ---- Step 1: synthetic unencrypted source DB ---------------------
        source_path = _make_temp_path("cka_sec02_src_")
        con = sqlite3.connect(source_path)
        try:
            con.execute(_SOURCE_SCHEMA)
            for rec in records:
                con.execute(
                    "INSERT OR REPLACE INTO sec02_synth_source "
                    "(record_id, label, payload) VALUES (?, ?, ?)",
                    rec,
                )
            con.commit()
        finally:
            con.close()
        result.synthetic_source_created = True
        result._source_pre_hash = _file_sha256(source_path)

        # ---- Step 2: read synthetic source records -----------------------
        con = sqlite3.connect(source_path)
        try:
            rows = list(con.execute(
                "SELECT record_id, label, payload FROM sec02_synth_source "
                "ORDER BY record_id"
            ).fetchall())
        finally:
            con.close()

        # ---- Step 3: encrypted target DB ---------------------------------
        target_path = _make_temp_path("cka_sec02_tgt_")
        records_copied = 0
        with EncryptedCKAStore(target_path, operator_key) as store:
            for rec in rows:
                store.insert_synthetic_record(rec[0], rec[1], rec[2])
                records_copied += 1
            # Step 4: verify correct-key read on the encrypted target.
            target_rows = store.fetch_all_synthetic()
        result.encrypted_target_created = True
        result.records_copied = records_copied
        result.correct_key_read_passed = (len(target_rows) == len(rows))

        # ---- Step 5: verify wrong-key fails ------------------------------
        wrong_key = "wrongkey_" + secrets.token_hex(16)
        if wrong_key == operator_key:
            wrong_key += "_x"
        result.wrong_key_failed = verify_wrong_key_fails(target_path, wrong_key)

        # ---- Step 6: verify plaintext absence in encrypted bytes ---------
        result.plaintext_absence_verified = verify_plaintext_absent(
            target_path, SYNTHETIC_FORBIDDEN_STRINGS,
        )

        # ---- Step 7: verify source DB unchanged --------------------------
        post_hash = _file_sha256(source_path)
        result.source_unchanged = (
            post_hash is not None
            and post_hash == result._source_pre_hash
        )

        result.rehearsal_performed = True
        return result
    finally:
        # Always delete temp files. Public reports never see these paths.
        _safe_unlink(source_path)
        _safe_unlink(target_path)
        # temp_files_staged stays False — the script never adds these to git.
        result.temp_files_staged = False


def rehearsal_passed(result: MigrationRehearsalResult) -> bool:
    """All-of check on a rehearsal result."""
    return all([
        result.rehearsal_performed,
        result.synthetic_source_created,
        result.encrypted_target_created,
        result.records_copied > 0,
        result.correct_key_read_passed,
        result.wrong_key_failed,
        result.plaintext_absence_verified,
        result.source_unchanged,
        not result.temp_files_staged,
    ])
