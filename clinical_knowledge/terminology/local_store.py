"""CKA-TERM-01 — local terminology index (SQLite).

A small metadata/coding index. By default the validation script and
tests use temp paths (`:memory:` or `tempfile`); a real on-disk store
is gitignored.

Hard rules:
- Does NOT modify the main MKB store.
- Does NOT promote facts to active tier.
- Does NOT clear DDI status.
- Does NOT make clinical recommendations.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

from clinical_knowledge.terminology.models import (
    TerminologyConcept,
    TerminologySystem,
    normalize_query,
)


_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS terminology_sources (
        source_id       TEXT PRIMARY KEY,
        safe_source_id  TEXT NOT NULL,
        system          TEXT NOT NULL,
        version         TEXT,
        license_confirmed INTEGER NOT NULL DEFAULT 0,
        created_at      TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS terminology_concepts (
        concept_id  TEXT PRIMARY KEY,
        system      TEXT NOT NULL,
        code        TEXT NOT NULL,
        display     TEXT NOT NULL,
        display_norm TEXT NOT NULL,
        version     TEXT,
        source_id   TEXT,
        active      INTEGER NOT NULL DEFAULT 1,
        synthetic   INTEGER NOT NULL DEFAULT 1
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS terminology_synonyms (
        concept_id  TEXT NOT NULL,
        synonym_norm TEXT NOT NULL,
        PRIMARY KEY (concept_id, synonym_norm)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS terminology_import_events (
        event_id    TEXT PRIMARY KEY,
        source_id   TEXT,
        ts          TEXT NOT NULL,
        event_type  TEXT NOT NULL,
        rows_seen   INTEGER NOT NULL DEFAULT 0,
        rows_imported INTEGER NOT NULL DEFAULT 0
    );
    """,
)

_INDEX_STATEMENTS = (
    "CREATE INDEX IF NOT EXISTS idx_term_concepts_display_norm "
    "ON terminology_concepts (display_norm);",
    "CREATE INDEX IF NOT EXISTS idx_term_concepts_system "
    "ON terminology_concepts (system);",
    "CREATE INDEX IF NOT EXISTS idx_term_synonyms_norm "
    "ON terminology_synonyms (synonym_norm);",
)


class LocalTerminologyStore:
    """SQLite-backed terminology index. Defaults to in-memory.

    Schema is versioned at v1; future blocks may extend it (no
    backward-incompatible mutations are made by TERM-01).
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self._mem_con: Optional[sqlite3.Connection] = None
        if db_path == ":memory:":
            self._mem_con = sqlite3.connect(":memory:")
        self._init_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        if self._mem_con is not None:
            self._mem_con.row_factory = sqlite3.Row
            yield self._mem_con
            self._mem_con.commit()
            return
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def _init_schema(self) -> None:
        with self._conn() as con:
            for stmt in _SCHEMA:
                con.execute(stmt)
            for stmt in _INDEX_STATEMENTS:
                con.execute(stmt)

    # ------------------------------------------------------------------
    # Source registration
    # ------------------------------------------------------------------

    def register_source(
        self,
        system: TerminologySystem,
        *,
        safe_source_id: Optional[str] = None,
        version: Optional[str] = None,
        license_confirmed: bool = False,
    ) -> str:
        sid = f"term_src_{uuid.uuid4().hex[:12]}"
        with self._conn() as con:
            con.execute(
                "INSERT INTO terminology_sources "
                "(source_id, safe_source_id, system, version, "
                "license_confirmed, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    sid,
                    safe_source_id or sid,
                    system.value,
                    version,
                    1 if license_confirmed else 0,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        return sid

    # ------------------------------------------------------------------
    # Concept ingestion
    # ------------------------------------------------------------------

    def add_concepts(
        self,
        concepts: Iterable[TerminologyConcept],
        source_id: Optional[str] = None,
    ) -> int:
        """Insert concepts (and their synonyms) into the index. Returns the count.

        The store does NOT promote any record into the MKB active tier.
        It is a metadata index only.
        """
        n = 0
        with self._conn() as con:
            for c in concepts:
                con.execute(
                    "INSERT OR REPLACE INTO terminology_concepts "
                    "(concept_id, system, code, display, display_norm, "
                    "version, source_id, active, synthetic) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        c.concept_id, c.system.value, c.code, c.display,
                        normalize_query(c.display),
                        c.version, source_id,
                        1 if c.active else 0,
                        1 if c.synthetic else 0,
                    ),
                )
                for syn in c.synonyms:
                    con.execute(
                        "INSERT OR IGNORE INTO terminology_synonyms "
                        "(concept_id, synonym_norm) VALUES (?, ?)",
                        (c.concept_id, normalize_query(syn)),
                    )
                n += 1
        return n

    def record_import_event(
        self,
        source_id: Optional[str],
        event_type: str,
        rows_seen: int,
        rows_imported: int,
    ) -> str:
        eid = f"term_evt_{uuid.uuid4().hex[:12]}"
        with self._conn() as con:
            con.execute(
                "INSERT INTO terminology_import_events "
                "(event_id, source_id, ts, event_type, rows_seen, rows_imported) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    eid, source_id,
                    datetime.now(timezone.utc).isoformat(),
                    event_type,
                    int(rows_seen), int(rows_imported),
                ),
            )
        return eid

    # ------------------------------------------------------------------
    # Read accessors
    # ------------------------------------------------------------------

    def count_concepts(self, system: Optional[TerminologySystem] = None) -> int:
        with self._conn() as con:
            if system is None:
                row = con.execute(
                    "SELECT count(*) AS n FROM terminology_concepts"
                ).fetchone()
            else:
                row = con.execute(
                    "SELECT count(*) AS n FROM terminology_concepts "
                    "WHERE system = ?",
                    (system.value,),
                ).fetchone()
        return int(row["n"]) if row else 0

    def fetch_concepts_by_norm(
        self,
        norm: str,
        systems: Optional[List[TerminologySystem]] = None,
        max_results: int = 10,
    ) -> List[TerminologyConcept]:
        """Return concepts whose display OR synonym normalizes to `norm`."""
        sys_filter = [s.value for s in (systems or [])]
        out: List[TerminologyConcept] = []
        with self._conn() as con:
            where = (
                "c.active = 1 AND "
                "(c.display_norm = ? OR c.concept_id IN "
                "(SELECT concept_id FROM terminology_synonyms WHERE synonym_norm = ?))"
            )
            params: List = [norm, norm]
            if sys_filter:
                placeholders = ",".join("?" * len(sys_filter))
                where += f" AND c.system IN ({placeholders})"
                params.extend(sys_filter)
            sql = (
                "SELECT c.concept_id, c.system, c.code, c.display, "
                "c.version, c.source_id, c.active, c.synthetic "
                "FROM terminology_concepts c WHERE " + where
                + " ORDER BY c.system, c.code LIMIT ?"
            )
            params.append(int(max_results))
            for row in con.execute(sql, params).fetchall():
                # Pull synonyms.
                syn_rows = con.execute(
                    "SELECT synonym_norm FROM terminology_synonyms "
                    "WHERE concept_id = ?",
                    (row["concept_id"],),
                ).fetchall()
                out.append(TerminologyConcept(
                    concept_id=row["concept_id"],
                    system=TerminologySystem(row["system"]),
                    code=row["code"],
                    display=row["display"],
                    synonyms=[r["synonym_norm"] for r in syn_rows],
                    version=row["version"],
                    source_safe_id=row["source_id"],
                    active=bool(row["active"]),
                    synthetic=bool(row["synthetic"]),
                ))
        return out

    def safe_public_summary(self) -> dict:
        with self._conn() as con:
            n_sources = con.execute(
                "SELECT count(*) AS n FROM terminology_sources"
            ).fetchone()["n"]
            n_concepts = con.execute(
                "SELECT count(*) AS n FROM terminology_concepts"
            ).fetchone()["n"]
            n_synonyms = con.execute(
                "SELECT count(*) AS n FROM terminology_synonyms"
            ).fetchone()["n"]
            n_events = con.execute(
                "SELECT count(*) AS n FROM terminology_import_events"
            ).fetchone()["n"]
        return {
            "in_memory": self._mem_con is not None,
            "sources_count": int(n_sources),
            "concepts_count": int(n_concepts),
            "synonyms_count": int(n_synonyms),
            "import_events_count": int(n_events),
        }
