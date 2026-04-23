"""MKB write adapter for deterministic execution results."""

from __future__ import annotations

from typing import Any

from app.schemas import MKBRecord


class MKBWriter:
    """Writes records through existing quality, SQL, and vector components."""

    def __init__(
        self,
        sql_store: Any | None = None,
        vector_store: Any | None = None,
        quality_gate: Any | None = None,
    ):
        self.sql_store = sql_store
        self.vector_store = vector_store
        self.quality_gate = quality_gate

    def write(self, records: list[MKBRecord], session_id: str = "") -> tuple[list[MKBRecord], list[MKBRecord]]:
        written: list[MKBRecord] = []
        queued: list[MKBRecord] = []

        for record in records:
            approved = True
            final = record
            reason = "approved"

            if self.quality_gate is not None:
                approved, reason, final = self.quality_gate.check(record, session_id=session_id)

            if not approved or final is None:
                if reason.startswith("Duplicate of ") or reason.startswith("Existing record retained:"):
                    continue
                queued.append(record)
                continue

            if final.requires_review or final.tier == "quarantined":
                queued.append(final)
                if self.sql_store is not None:
                    self.sql_store.write_record(final, session_id=session_id)
                continue

            if self.sql_store is not None:
                self.sql_store.write_record(final, session_id=session_id)
            if self.vector_store is not None:
                self.vector_store.add_record(final)
            written.append(final)

        return written, queued
