"""CKA-TERM-01C transaction wrapper for synthetic terminology imports."""
from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Iterable

from clinical_knowledge.terminology.local_store import LocalTerminologyStore
from clinical_knowledge.terminology.models import (
    TerminologyConcept,
    TerminologySystem,
    normalize_query,
)


class TerminologyImportTransaction:
    """Explicit transaction boundary for a LocalTerminologyStore.

    TERM-01C uses this only for synthetic/temp imports. It intentionally avoids
    production index paths and does not alter the main MKB store.
    """

    def __init__(self, store: LocalTerminologyStore) -> None:
        self.store = store
        self._con: sqlite3.Connection | None = None
        self._owns_connection = False
        self.started = False
        self.committed = False
        self.rolled_back = False

    def __enter__(self) -> "TerminologyImportTransaction":
        self.begin()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.rollback()
            return
        if self.started and not self.committed and not self.rolled_back:
            self.commit()

    def begin(self) -> None:
        if self.started:
            return
        if self.store._mem_con is not None:  # noqa: SLF001 - same package transaction adapter
            self._con = self.store._mem_con  # noqa: SLF001
        else:
            self._con = sqlite3.connect(self.store.db_path)
            self._owns_connection = True
        self._con.row_factory = sqlite3.Row
        self._con.execute("BEGIN")
        self.started = True

    def write_source_manifest(
        self,
        system: TerminologySystem,
        *,
        source_safe_id: str,
        version: str = "synthetic-test-1",
        license_confirmed: bool = True,
    ) -> str:
        con = self._require_connection()
        source_id = f"term_src_{uuid.uuid4().hex[:12]}"
        con.execute(
            "INSERT INTO terminology_sources "
            "(source_id, safe_source_id, system, version, license_confirmed, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                source_id,
                source_safe_id,
                system.value,
                version,
                1 if license_confirmed else 0,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        return source_id

    def write_concepts(self, concepts: Iterable[TerminologyConcept], source_id: str) -> int:
        con = self._require_connection()
        imported = 0
        for concept in concepts:
            con.execute(
                "INSERT OR REPLACE INTO terminology_concepts "
                "(concept_id, system, code, display, display_norm, version, source_id, active, synthetic) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    concept.concept_id,
                    concept.system.value,
                    concept.code,
                    concept.display,
                    normalize_query(concept.display),
                    concept.version,
                    source_id,
                    1 if concept.active else 0,
                    1 if concept.synthetic else 0,
                ),
            )
            for synonym in concept.synonyms:
                con.execute(
                    "INSERT OR IGNORE INTO terminology_synonyms "
                    "(concept_id, synonym_norm) VALUES (?, ?)",
                    (concept.concept_id, normalize_query(synonym)),
                )
            imported += 1
        return imported

    def write_import_audit_event(
        self,
        source_id: str,
        *,
        event_type: str,
        rows_seen: int,
        rows_imported: int,
    ) -> str:
        con = self._require_connection()
        event_id = f"term_evt_{uuid.uuid4().hex[:12]}"
        con.execute(
            "INSERT INTO terminology_import_events "
            "(event_id, source_id, ts, event_type, rows_seen, rows_imported) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                event_id,
                source_id,
                datetime.now(timezone.utc).isoformat(),
                event_type,
                int(rows_seen),
                int(rows_imported),
            ),
        )
        return event_id

    def commit(self) -> None:
        if self._con is not None and self.started and not self.committed:
            self._con.commit()
            self.committed = True
        self._close_owned_connection()

    def rollback(self) -> None:
        if self._con is not None and self.started and not self.rolled_back:
            self._con.rollback()
            self.rolled_back = True
        self._close_owned_connection()

    def _require_connection(self) -> sqlite3.Connection:
        if self._con is None:
            raise RuntimeError("transaction_not_started")
        return self._con

    def _close_owned_connection(self) -> None:
        if self._owns_connection and self._con is not None:
            self._con.close()
        self._con = None
