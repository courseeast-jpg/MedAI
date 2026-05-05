"""Local SQLite store abstraction for CKA-B01.

Uses stdlib sqlite3. SQLCipher encryption is NOT active in this block.
The report field "sqlcipher_encryption_active" is explicitly set to false.
"encryption_boundary_ready" is true (schema is ready for future encryption).

Do NOT claim SQLCipher is active unless real SQLCipher integration is done.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List, Optional

from clinical_knowledge.models import (
    DDIStatus,
    KnowledgeTier,
    LedgerEvent,
    LedgerEventType,
    MKBRecord,
    RecordStatus,
    SourceType,
    TrustLevel,
)

SQLCIPHER_ENCRYPTION_ACTIVE = False
ENCRYPTION_BOUNDARY_READY = True

_CREATE_RECORDS = """
CREATE TABLE IF NOT EXISTS mkb_records (
    record_id       TEXT PRIMARY KEY,
    safe_record_id  TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    fact_type       TEXT NOT NULL,
    entity_text     TEXT NOT NULL,
    structured      TEXT NOT NULL DEFAULT '{}',
    specialty       TEXT NOT NULL DEFAULT 'general',
    source_type     TEXT NOT NULL,
    source_ref      TEXT NOT NULL DEFAULT '',
    trust_level     INTEGER NOT NULL,
    tier            TEXT NOT NULL,
    status          TEXT NOT NULL,
    confidence      REAL NOT NULL DEFAULT 0.0,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    ddi_checked     INTEGER NOT NULL DEFAULT 0,
    ddi_status      TEXT NOT NULL DEFAULT 'not_checked',
    ddi_findings    TEXT NOT NULL DEFAULT '[]',
    extraction_method TEXT NOT NULL DEFAULT 'manual',
    resolution_id   TEXT,
    promotion_history TEXT NOT NULL DEFAULT '[]',
    requires_review INTEGER NOT NULL DEFAULT 0
)
"""

_CREATE_LEDGER = """
CREATE TABLE IF NOT EXISTS ledger_events (
    event_id        TEXT PRIMARY KEY,
    event_type      TEXT NOT NULL,
    record_id       TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    actor           TEXT NOT NULL DEFAULT 'system',
    reason          TEXT NOT NULL DEFAULT '',
    details         TEXT NOT NULL DEFAULT '{}',
    safe_public_details TEXT NOT NULL DEFAULT '{}'
)
"""


@contextmanager
def _conn(db_path: str, _persistent: Optional[sqlite3.Connection] = None) -> Generator[sqlite3.Connection, None, None]:
    if _persistent is not None:
        _persistent.row_factory = sqlite3.Row
        yield _persistent
        _persistent.commit()
    else:
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()


class MKBStore:
    """Local SQLite-backed store for MKBRecords and ledger events."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        # Keep a persistent connection for :memory: so schema survives across calls
        self._mem_con: Optional[sqlite3.Connection] = None
        if db_path == ":memory:":
            self._mem_con = sqlite3.connect(":memory:")
        self._init_schema()

    def _init_schema(self) -> None:
        with _conn(self.db_path, self._mem_con) as con:
            con.execute(_CREATE_RECORDS)
            con.execute(_CREATE_LEDGER)

    # ------------------------------------------------------------------
    # Record operations
    # ------------------------------------------------------------------

    def insert_record(self, record: MKBRecord, ledger_event: Optional[LedgerEvent] = None) -> None:
        with _conn(self.db_path, self._mem_con) as con:
            con.execute(
                """INSERT INTO mkb_records VALUES (
                    ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                )""",
                (
                    record.record_id,
                    record.safe_record_id,
                    record.session_id,
                    record.fact_type,
                    record.entity_text,
                    json.dumps(record.structured),
                    record.specialty,
                    record.source_type.value if isinstance(record.source_type, SourceType) else record.source_type,
                    record.source_ref,
                    record.trust_level.value if isinstance(record.trust_level, TrustLevel) else record.trust_level,
                    record.tier.value if isinstance(record.tier, KnowledgeTier) else record.tier,
                    record.status.value if isinstance(record.status, RecordStatus) else record.status,
                    record.confidence,
                    record.created_at,
                    record.updated_at,
                    int(record.ddi_checked),
                    record.ddi_status.value if isinstance(record.ddi_status, DDIStatus) else record.ddi_status,
                    json.dumps(record.ddi_findings),
                    record.extraction_method,
                    record.resolution_id,
                    json.dumps(record.promotion_history),
                    int(record.requires_review),
                ),
            )
            if ledger_event:
                self._insert_ledger_event_con(con, ledger_event)

    def update_record_tier(
        self,
        record_id: str,
        new_tier: KnowledgeTier,
        updated_at: str,
        ledger_event: Optional[LedgerEvent] = None,
    ) -> None:
        requires_review = 1 if new_tier in (KnowledgeTier.QUARANTINED, KnowledgeTier.SUPERSEDED) else 0
        with _conn(self.db_path, self._mem_con) as con:
            con.execute(
                "UPDATE mkb_records SET tier=?, requires_review=?, updated_at=? WHERE record_id=?",
                (new_tier.value, requires_review, updated_at, record_id),
            )
            if ledger_event:
                self._insert_ledger_event_con(con, ledger_event)

    def update_record_status(
        self,
        record_id: str,
        new_status: RecordStatus,
        updated_at: str,
        ledger_event: Optional[LedgerEvent] = None,
    ) -> None:
        with _conn(self.db_path, self._mem_con) as con:
            con.execute(
                "UPDATE mkb_records SET status=?, updated_at=? WHERE record_id=?",
                (new_status.value, updated_at, record_id),
            )
            if ledger_event:
                self._insert_ledger_event_con(con, ledger_event)

    def fetch_by_record_id(self, record_id: str) -> Optional[dict]:
        with _conn(self.db_path, self._mem_con) as con:
            row = con.execute(
                "SELECT * FROM mkb_records WHERE record_id=?", (record_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_active(self) -> List[dict]:
        with _conn(self.db_path, self._mem_con) as con:
            rows = con.execute(
                "SELECT * FROM mkb_records WHERE tier=?", (KnowledgeTier.ACTIVE.value,)
            ).fetchall()
        return [dict(r) for r in rows]

    def list_hypothesis(self) -> List[dict]:
        with _conn(self.db_path, self._mem_con) as con:
            rows = con.execute(
                "SELECT * FROM mkb_records WHERE tier=?", (KnowledgeTier.HYPOTHESIS.value,)
            ).fetchall()
        return [dict(r) for r in rows]

    def list_quarantined(self) -> List[dict]:
        with _conn(self.db_path, self._mem_con) as con:
            rows = con.execute(
                "SELECT * FROM mkb_records WHERE tier=?", (KnowledgeTier.QUARANTINED.value,)
            ).fetchall()
        return [dict(r) for r in rows]

    def list_superseded(self) -> List[dict]:
        with _conn(self.db_path, self._mem_con) as con:
            rows = con.execute(
                "SELECT * FROM mkb_records WHERE tier=?", (KnowledgeTier.SUPERSEDED.value,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Ledger operations
    # ------------------------------------------------------------------

    def _insert_ledger_event_con(self, con: sqlite3.Connection, event: LedgerEvent) -> None:
        con.execute(
            "INSERT INTO ledger_events VALUES (?,?,?,?,?,?,?,?)",
            (
                event.event_id,
                event.event_type.value if isinstance(event.event_type, LedgerEventType) else event.event_type,
                event.record_id,
                event.timestamp,
                event.actor,
                event.reason,
                json.dumps(event.details),
                json.dumps(event.safe_public_details),
            ),
        )

    def append_ledger_event(self, event: LedgerEvent) -> None:
        with _conn(self.db_path, self._mem_con) as con:
            self._insert_ledger_event_con(con, event)

    def read_ledger_events(self, record_id: Optional[str] = None) -> List[dict]:
        with _conn(self.db_path, self._mem_con) as con:
            if record_id:
                rows = con.execute(
                    "SELECT * FROM ledger_events WHERE record_id=? ORDER BY timestamp",
                    (record_id,),
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT * FROM ledger_events ORDER BY timestamp"
                ).fetchall()
        return [dict(r) for r in rows]

    def count_ledger_events(self) -> int:
        with _conn(self.db_path, self._mem_con) as con:
            return con.execute("SELECT COUNT(*) FROM ledger_events").fetchone()[0]
