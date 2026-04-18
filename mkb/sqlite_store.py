"""
MedAI v1.1 — MKB SQLite Store (Track A)
Encrypted with SQLCipher. Falls back to standard SQLite if SQLCipher unavailable.
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from loguru import logger

from app.config import DB_PATH, TIER_ACTIVE, TIER_HYPOTHESIS, TIER_QUARANTINED
from app.schemas import MKBRecord, LedgerEvent

try:
    from sqlcipher3 import dbapi2 as sqlite_backend
    ENCRYPTED = True
    logger.info("SQLCipher encryption available")
except ImportError:
    import sqlite3 as sqlite_backend
    ENCRYPTED = False
    logger.warning("SQLCipher not available — using unencrypted SQLite. Install sqlcipher3 for encryption.")


DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    id                TEXT PRIMARY KEY,
    fact_type         TEXT NOT NULL,
    content           TEXT NOT NULL,
    structured_json   TEXT DEFAULT '{}',
    specialty         TEXT DEFAULT 'general',
    source_type       TEXT NOT NULL,
    source_name       TEXT DEFAULT '',
    source_url        TEXT,
    trust_level       INTEGER DEFAULT 3,
    confidence        REAL DEFAULT 0.5,
    status            TEXT DEFAULT 'active',
    tier              TEXT DEFAULT 'active',
    ddi_checked       INTEGER DEFAULT 0,
    ddi_status        TEXT,
    ddi_findings_json TEXT DEFAULT '[]',
    extraction_method TEXT DEFAULT 'claude',
    resolution_id     TEXT,
    requires_review   INTEGER DEFAULT 0,
    first_recorded    TEXT NOT NULL,
    last_confirmed    TEXT NOT NULL,
    linked_ids_json   TEXT DEFAULT '[]',
    chunk_ids_json    TEXT DEFAULT '[]',
    tags_json         TEXT DEFAULT '[]',
    session_id        TEXT DEFAULT '',
    promotion_history TEXT DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS ledger (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type      TEXT NOT NULL,
    record_id       TEXT,
    source_type     TEXT,
    previous_value  TEXT,
    details_json    TEXT DEFAULT '{}',
    timestamp       TEXT NOT NULL,
    session_id      TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_records_specialty ON records(specialty);
CREATE INDEX IF NOT EXISTS idx_records_tier ON records(tier);
CREATE INDEX IF NOT EXISTS idx_records_fact_type ON records(fact_type);
CREATE INDEX IF NOT EXISTS idx_records_status ON records(status);
CREATE INDEX IF NOT EXISTS idx_records_trust_level ON records(trust_level);
CREATE INDEX IF NOT EXISTS idx_ledger_event_type ON ledger(event_type);
CREATE INDEX IF NOT EXISTS idx_ledger_record_id ON ledger(record_id);
"""


class SQLiteStore:
    def __init__(self, db_path: Path = DB_PATH, encryption_key: str = ""):
        self.db_path = db_path
        self.encryption_key = encryption_key
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self):
        conn = sqlite_backend.connect(str(self.db_path))
        if ENCRYPTED and self.encryption_key:
            conn.execute(f"PRAGMA key='{self.encryption_key}'")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.executescript(DB_SCHEMA)
        logger.info(f"SQLite store initialized at {self.db_path}")

    # ── WRITE ──────────────────────────────────────────────────────────────

    def write_record(self, record: MKBRecord, session_id: str = "") -> str:
        record.session_id = session_id
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO records VALUES (
                    :id, :fact_type, :content, :structured_json, :specialty,
                    :source_type, :source_name, :source_url, :trust_level, :confidence,
                    :status, :tier, :ddi_checked, :ddi_status, :ddi_findings_json,
                    :extraction_method, :resolution_id, :requires_review,
                    :first_recorded, :last_confirmed, :linked_ids_json,
                    :chunk_ids_json, :tags_json, :session_id, :promotion_history
                )
            """, {
                "id": record.id,
                "fact_type": record.fact_type,
                "content": record.content,
                "structured_json": json.dumps(record.structured),
                "specialty": record.specialty,
                "source_type": record.source_type,
                "source_name": record.source_name,
                "source_url": record.source_url,
                "trust_level": record.trust_level,
                "confidence": record.confidence,
                "status": record.status,
                "tier": record.tier,
                "ddi_checked": int(record.ddi_checked),
                "ddi_status": record.ddi_status,
                "ddi_findings_json": json.dumps(record.ddi_findings),
                "extraction_method": record.extraction_method,
                "resolution_id": record.resolution_id,
                "requires_review": int(record.requires_review),
                "first_recorded": record.first_recorded.isoformat(),
                "last_confirmed": record.last_confirmed.isoformat(),
                "linked_ids_json": json.dumps(record.linked_to),
                "chunk_ids_json": json.dumps(record.chunk_ids),
                "tags_json": json.dumps(record.tags),
                "session_id": record.session_id,
                "promotion_history": json.dumps(record.promotion_history),
            })
        logger.debug(f"Written record {record.id} tier={record.tier} trust={record.trust_level}")
        return record.id

    def write_ledger(self, event: LedgerEvent) -> int:
        with self._get_conn() as conn:
            cur = conn.execute("""
                INSERT INTO ledger (event_type, record_id, source_type, previous_value,
                                    details_json, timestamp, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_type,
                event.record_id,
                event.source_type,
                json.dumps(event.previous_value) if event.previous_value else None,
                json.dumps(event.details),
                event.timestamp.isoformat(),
                event.session_id,
            ))
            return cur.lastrowid

    # ── READ ───────────────────────────────────────────────────────────────

    def get_record(self, record_id: str) -> Optional[MKBRecord]:
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM records WHERE id=?", (record_id,)).fetchone()
        return self._row_to_record(row) if row else None

    def get_active_medications(self) -> List[MKBRecord]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM records WHERE fact_type='medication' AND status='active' AND tier='active'"
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_active_diagnoses(self) -> List[MKBRecord]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM records WHERE fact_type='diagnosis' AND status='active' AND tier='active'"
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_recent_conflicts(self, days: int = 90) -> List[MKBRecord]:
        cutoff = datetime.utcnow().isoformat()[:10]
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM records WHERE status='conflicted' AND last_confirmed >= ?",
                (cutoff,)
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_records_requiring_review(self) -> List[MKBRecord]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM records WHERE requires_review=1"
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_by_specialty(self, specialty: str, tier: Optional[str] = None) -> List[MKBRecord]:
        query = "SELECT * FROM records WHERE specialty=? AND status='active'"
        params = [specialty]
        if tier:
            query += " AND tier=?"
            params.append(tier)
        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_record(r) for r in rows]

    def update_status(self, record_id: str, status: str, tier: Optional[str] = None):
        if tier:
            with self._get_conn() as conn:
                conn.execute(
                    "UPDATE records SET status=?, tier=?, last_confirmed=? WHERE id=?",
                    (status, tier, datetime.utcnow().isoformat(), record_id)
                )
        else:
            with self._get_conn() as conn:
                conn.execute(
                    "UPDATE records SET status=?, last_confirmed=? WHERE id=?",
                    (status, datetime.utcnow().isoformat(), record_id)
                )

    def count_records(self) -> dict:
        with self._get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
            active = conn.execute("SELECT COUNT(*) FROM records WHERE tier='active'").fetchone()[0]
            hyp = conn.execute("SELECT COUNT(*) FROM records WHERE tier='hypothesis'").fetchone()[0]
            quar = conn.execute("SELECT COUNT(*) FROM records WHERE tier='quarantined'").fetchone()[0]
        return {"total": total, "active": active, "hypothesis": hyp, "quarantined": quar}

    # ── HELPERS ────────────────────────────────────────────────────────────

    def _row_to_record(self, row) -> MKBRecord:
        d = dict(row)
        return MKBRecord(
            id=d["id"],
            fact_type=d["fact_type"],
            content=d["content"],
            structured=json.loads(d["structured_json"] or "{}"),
            specialty=d["specialty"],
            source_type=d["source_type"],
            source_name=d["source_name"] or "",
            source_url=d["source_url"],
            trust_level=d["trust_level"],
            confidence=d["confidence"],
            status=d["status"],
            tier=d["tier"],
            ddi_checked=bool(d["ddi_checked"]),
            ddi_status=d["ddi_status"],
            ddi_findings=json.loads(d["ddi_findings_json"] or "[]"),
            extraction_method=d["extraction_method"] or "claude",
            resolution_id=d["resolution_id"],
            requires_review=bool(d["requires_review"]),
            first_recorded=datetime.fromisoformat(d["first_recorded"]),
            last_confirmed=datetime.fromisoformat(d["last_confirmed"]),
            linked_to=json.loads(d["linked_ids_json"] or "[]"),
            chunk_ids=json.loads(d["chunk_ids_json"] or "[]"),
            tags=json.loads(d["tags_json"] or "[]"),
            session_id=d["session_id"] or "",
            promotion_history=json.loads(d["promotion_history"] or "[]"),
        )
