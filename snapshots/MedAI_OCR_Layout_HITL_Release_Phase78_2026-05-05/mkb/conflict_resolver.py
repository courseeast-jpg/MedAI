"""
MedAI — Conflict Resolver (Phase 5)

Persists quarantined conflicts to a dedicated SQLite table and applies user
resolutions. Works alongside ``mkb.sqlite_store.SQLiteStore`` and the truth
resolution engine. The conflicts table is created lazily on first use so this
module never blocks system startup if the main schema is already migrated.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from app.config import DB_PATH, TIER_ACTIVE, TIER_QUARANTINED, TIER_SUPERSEDED


_CONFLICT_SCHEMA = """
CREATE TABLE IF NOT EXISTS conflicts (
    id              TEXT PRIMARY KEY,
    fact1_id        TEXT,
    fact2_id        TEXT,
    fact1_snapshot  TEXT NOT NULL,
    fact2_snapshot  TEXT NOT NULL,
    conflict_type   TEXT NOT NULL,
    severity        TEXT NOT NULL DEFAULT 'medium',
    reason          TEXT DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'pending',     -- pending|resolved|dismissed
    resolution      TEXT,                                 -- JSON of the user's choice
    resolution_notes TEXT,
    created_at      TEXT NOT NULL,
    resolved_at     TEXT,
    session_id      TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_conflicts_status    ON conflicts(status);
CREATE INDEX IF NOT EXISTS idx_conflicts_severity  ON conflicts(severity);
CREATE INDEX IF NOT EXISTS idx_conflicts_fact1     ON conflicts(fact1_id);
CREATE INDEX IF NOT EXISTS idx_conflicts_fact2     ON conflicts(fact2_id);

CREATE TABLE IF NOT EXISTS conflict_audit_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    conflict_id  TEXT NOT NULL,
    action       TEXT NOT NULL,
    payload      TEXT DEFAULT '{}',
    timestamp    TEXT NOT NULL
);
"""

_VALID_CHOICES = {"fact1", "fact2", "both", "merge", "neither"}
_VALID_SEVERITIES = {"low", "medium", "high", "critical"}


class ConflictResolver:
    """Stores conflicts and applies user resolutions."""

    def __init__(self, db_path: Path = DB_PATH, sql_store=None):
        """
        :param db_path: path to the SQLite database (shares file with SQLiteStore).
        :param sql_store: optional ``SQLiteStore`` used to update record tiers.
                          Required for ``resolve_conflict`` to flip record status.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._sql_store = sql_store
        self._init_db()

    # ── schema ──────────────────────────────────────────────────────────────

    def _conn(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_CONFLICT_SCHEMA)
        logger.info(f"Conflict store initialized at {self.db_path}")

    # ── write ───────────────────────────────────────────────────────────────

    def quarantine_conflict(
        self,
        fact1: dict[str, Any],
        fact2: dict[str, Any],
        conflict_type: str,
        severity: str = "medium",
        reason: str = "",
        session_id: str = "",
    ) -> str:
        """
        Store a conflict for user review.
        1. Create conflict record with both fact snapshots.
        2. Mark both facts as ``status='quarantined'``.
        3. Severity determines UI prominence (``low|medium|high|critical``).
        4. Audit log entry is always recorded.
        """
        if severity not in _VALID_SEVERITIES:
            raise ValueError(f"severity must be one of {_VALID_SEVERITIES}, got {severity!r}")

        conflict_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO conflicts (
                    id, fact1_id, fact2_id, fact1_snapshot, fact2_snapshot,
                    conflict_type, severity, reason, status, created_at, session_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    conflict_id,
                    fact1.get("id"),
                    fact2.get("id"),
                    json.dumps(_json_safe(fact1)),
                    json.dumps(_json_safe(fact2)),
                    conflict_type,
                    severity,
                    reason,
                    now,
                    session_id,
                ),
            )
            self._audit(conn, conflict_id, "quarantined", {
                "conflict_type": conflict_type,
                "severity": severity,
                "reason": reason,
            })

        self._mark_record(fact1.get("id"), status="quarantined", tier=TIER_QUARANTINED)
        self._mark_record(fact2.get("id"), status="quarantined", tier=TIER_QUARANTINED)

        logger.warning(
            f"Quarantined conflict {conflict_id} ({conflict_type}, severity={severity}): {reason}"
        )
        return conflict_id

    # ── resolve ─────────────────────────────────────────────────────────────

    def resolve_conflict(self, conflict_id: str, resolution: dict[str, Any]) -> None:
        """
        Apply the user's resolution.

        resolution = {
            'choice':       'fact1' | 'fact2' | 'both' | 'merge' | 'neither',
            'merged_value': str (only if choice == 'merge'),
            'reasoning':    str,
            'notes':        str,
        }
        """
        choice = (resolution or {}).get("choice")
        if choice not in _VALID_CHOICES:
            raise ValueError(f"choice must be one of {_VALID_CHOICES}, got {choice!r}")

        conflict = self.get_conflict(conflict_id)
        if conflict is None:
            raise KeyError(f"Unknown conflict_id: {conflict_id}")
        if conflict["status"] != "pending":
            raise ValueError(f"Conflict {conflict_id} already {conflict['status']}")

        fact1_id = conflict["fact1_id"]
        fact2_id = conflict["fact2_id"]

        if choice == "fact1":
            self._mark_record(fact1_id, status="active", tier=TIER_ACTIVE)
            self._mark_record(fact2_id, status="rejected", tier=TIER_SUPERSEDED)
        elif choice == "fact2":
            self._mark_record(fact1_id, status="rejected", tier=TIER_SUPERSEDED)
            self._mark_record(fact2_id, status="active", tier=TIER_ACTIVE)
        elif choice == "both":
            self._mark_record(fact1_id, status="active", tier=TIER_ACTIVE)
            self._mark_record(fact2_id, status="active", tier=TIER_ACTIVE)
        elif choice == "neither":
            self._mark_record(fact1_id, status="rejected", tier=TIER_SUPERSEDED)
            self._mark_record(fact2_id, status="rejected", tier=TIER_SUPERSEDED)
        elif choice == "merge":
            # Write a new merged record via SQLiteStore if available; otherwise
            # stash the merged payload in the audit log for later reconciliation.
            merged_value = resolution.get("merged_value")
            merged_id = self._write_merged_record(conflict, merged_value)
            self._mark_record(fact1_id, status="merged", tier=TIER_SUPERSEDED)
            self._mark_record(fact2_id, status="merged", tier=TIER_SUPERSEDED)
            resolution = {**resolution, "merged_record_id": merged_id}

        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE conflicts
                   SET status='resolved',
                       resolution=?,
                       resolution_notes=?,
                       resolved_at=?
                 WHERE id=?
                """,
                (
                    json.dumps(_json_safe(resolution)),
                    resolution.get("notes") or resolution.get("reasoning") or "",
                    now,
                    conflict_id,
                ),
            )
            self._audit(conn, conflict_id, f"resolved:{choice}", resolution)

        logger.info(f"Resolved conflict {conflict_id} as {choice}")

    # ── read ────────────────────────────────────────────────────────────────

    def get_conflict(self, conflict_id: str) -> Optional[dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM conflicts WHERE id=?", (conflict_id,)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def list_pending(self, severity: Optional[str] = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM conflicts WHERE status='pending'"
        params: list[Any] = []
        if severity:
            query += " AND severity=?"
            params.append(severity)
        query += " ORDER BY CASE severity "\
                 "WHEN 'critical' THEN 0 WHEN 'high' THEN 1 "\
                 "WHEN 'medium' THEN 2 ELSE 3 END, created_at DESC"
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def count_pending(self) -> int:
        with self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM conflicts WHERE status='pending'"
            ).fetchone()[0]

    # ── helpers ─────────────────────────────────────────────────────────────

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        for key in ("fact1_snapshot", "fact2_snapshot", "resolution"):
            if d.get(key):
                try:
                    d[key] = json.loads(d[key])
                except json.JSONDecodeError:
                    pass
        return d

    def _audit(self, conn, conflict_id: str, action: str, payload: dict[str, Any]) -> None:
        conn.execute(
            """
            INSERT INTO conflict_audit_log (conflict_id, action, payload, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            (
                conflict_id,
                action,
                json.dumps(_json_safe(payload)),
                datetime.utcnow().isoformat(),
            ),
        )

    def _mark_record(self, record_id: Optional[str], status: str, tier: str) -> None:
        if not record_id or self._sql_store is None:
            return
        try:
            self._sql_store.update_status(record_id, status=status, tier=tier)
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Could not update record {record_id}: {exc}")

    def _write_merged_record(
        self,
        conflict: dict[str, Any],
        merged_value: Any,
    ) -> Optional[str]:
        """Best-effort: create a new MKBRecord representing the merged fact."""
        if self._sql_store is None:
            return None
        try:
            from app.schemas import MKBRecord
            f1 = conflict.get("fact1_snapshot") or {}
            f2 = conflict.get("fact2_snapshot") or {}
            content = (
                f"Merged: {f1.get('entity_name') or f1.get('content') or ''} / "
                f"{f2.get('entity_name') or f2.get('content') or ''}"
            )
            merged = MKBRecord(
                fact_type=f1.get("fact_type") or f2.get("fact_type") or "note",
                content=f"{content} → {merged_value}" if merged_value is not None else content,
                structured={
                    "merged_from": [f1.get("id"), f2.get("id")],
                    "merged_value": merged_value,
                },
                source_type="manual",
                source_name="conflict_resolution",
                trust_level=2,
                confidence=0.85,
                status="active",
                tier=TIER_ACTIVE,
            )
            self._sql_store.write_record(merged)
            return merged.id
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Merged record write failed: {exc}")
            return None


# ── utils ────────────────────────────────────────────────────────────────────

def _json_safe(obj: Any) -> Any:
    """Coerce datetime/date/etc. to strings so json.dumps succeeds."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)
    if hasattr(obj, "model_dump"):
        try:
            return _json_safe(obj.model_dump())
        except Exception:
            return str(obj)
    return obj
