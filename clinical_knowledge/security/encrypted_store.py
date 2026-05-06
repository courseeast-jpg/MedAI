"""CKA-SEC-01 — encrypted store adapter (synthetic-only).

EncryptedCKAStore is a *parallel* adapter for proving SQLCipher
encryption works on a synthetic schema. It does NOT replace the main
CKA MKBStore. It does NOT migrate real data.

Safety invariants:
- empty key is refused at construction time
- the encryption key is never logged, returned, or serialized in
  safe_public_summary
- all schema/insert/read operations apply PRAGMA key BEFORE any other
  query (SQLCipher requirement)
- if no SQLCipher-capable provider is available, instantiation raises
  EncryptedStoreError("provider_unavailable") rather than silently
  using plaintext
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple

from clinical_knowledge.security.sqlcipher_provider import (
    SQLCipherProviderStatus,
    detect_sqlcipher_provider,
)


class EncryptedStoreError(Exception):
    """Raised when the encrypted-store adapter cannot operate safely."""


_SYNTHETIC_SCHEMA = """
CREATE TABLE IF NOT EXISTS sec01_synthetic_records (
    record_id   TEXT PRIMARY KEY,
    label       TEXT NOT NULL,
    payload     TEXT NOT NULL
)
"""


def _quote_pragma_key(key: str) -> str:
    """Quote a string for the PRAGMA key statement.

    SQLCipher accepts ``PRAGMA key = 'literal'``. We escape any single
    quotes by doubling them (standard SQL string escape). The key is
    never logged, never returned, and is held only as a local variable
    during execution.
    """
    return "'" + key.replace("'", "''") + "'"


class EncryptedCKAStore:
    """Synthetic encrypted store adapter (parallel to MKBStore).

    Construction:
        EncryptedCKAStore(db_path, encryption_key)

    The key is held only on the open connection's PRAGMA state. It is
    not stored as an attribute. Closing the store discards the
    connection.
    """

    def __init__(
        self,
        db_path: str,
        encryption_key: str,
        *,
        provider_status: Optional[SQLCipherProviderStatus] = None,
    ) -> None:
        if not isinstance(encryption_key, str):
            raise EncryptedStoreError("encryption_key_must_be_str")
        if encryption_key == "":
            raise EncryptedStoreError("empty_encryption_key_refused")
        if not isinstance(db_path, str) or db_path == "":
            raise EncryptedStoreError("db_path_required")

        status = provider_status if provider_status is not None else detect_sqlcipher_provider()
        if not status.available or status.module is None:
            raise EncryptedStoreError("provider_unavailable")

        self._db_path = db_path
        self._provider_name = status.provider_name
        self._cipher_version = status.cipher_version
        self._module = status.module
        self._con = None    # type: ignore[assignment]
        self._open_with_key(encryption_key)
        self._init_schema()

    # ------------------------------------------------------------------
    # Internal: open + key
    # ------------------------------------------------------------------
    def _open_with_key(self, encryption_key: str) -> None:
        self._con = self._module.connect(self._db_path)    # type: ignore[union-attr]
        # Apply the SQLCipher key BEFORE any other query.
        # Use parameterless PRAGMA (parameter binding is not allowed for PRAGMA).
        self._con.execute(f"PRAGMA key = {_quote_pragma_key(encryption_key)};")
        # Touch a metadata query to force the key to be evaluated.
        self._con.execute("SELECT count(*) FROM sqlite_master;").fetchone()
        # encryption_key goes out of scope here.

    def _init_schema(self) -> None:
        assert self._con is not None
        self._con.execute(_SYNTHETIC_SCHEMA)
        self._con.commit()

    # ------------------------------------------------------------------
    # Public synthetic operations
    # ------------------------------------------------------------------
    def insert_synthetic_record(
        self,
        record_id: str,
        label: str,
        payload: str,
    ) -> None:
        """Insert a synthetic record. payload must be synthetic test data only."""
        assert self._con is not None
        self._con.execute(
            "INSERT OR REPLACE INTO sec01_synthetic_records "
            "(record_id, label, payload) VALUES (?, ?, ?)",
            (record_id, label, payload),
        )
        self._con.commit()

    def fetch_all_synthetic(self) -> List[Tuple[str, str, str]]:
        assert self._con is not None
        cur = self._con.execute(
            "SELECT record_id, label, payload FROM sec01_synthetic_records "
            "ORDER BY record_id"
        )
        return [tuple(row) for row in cur.fetchall()]

    def fetch_by_id(self, record_id: str) -> Optional[Tuple[str, str, str]]:
        assert self._con is not None
        cur = self._con.execute(
            "SELECT record_id, label, payload FROM sec01_synthetic_records "
            "WHERE record_id = ?",
            (record_id,),
        )
        row = cur.fetchone()
        return tuple(row) if row is not None else None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._con is not None:
            try:
                self._con.close()
            except Exception:    # noqa: BLE001
                pass
            self._con = None

    def __enter__(self) -> "EncryptedCKAStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public summary (NEVER includes the key)
    # ------------------------------------------------------------------
    @property
    def db_path(self) -> str:
        return self._db_path

    def safe_public_summary(self) -> dict:
        """Public-report-safe summary. Never includes the encryption key
        or full filesystem path; only a stable hash-like length marker.
        """
        return {
            "provider_name": self._provider_name,
            "cipher_version": self._cipher_version,
            "schema_initialized": True,
            "encryption_key_logged": False,
            "encryption_key_in_summary": False,
        }
