"""
MedAI MKB — Schema v2 migration.

Additive migration:
    * Creates new v2 tables (facts, timeline, conflicts, family_history,
      provenance, audit_log, terminology_aliases, documents) if absent.
    * Back-fills v1 ``records`` rows into ``facts`` once, driven by a
      ``migrations`` bookkeeping table so reruns are idempotent.

Run:
    python -m mkb.migrate_to_v2                         # uses DB_PATH
    python -m mkb.migrate_to_v2 --db-path /path/to.db   # custom
    python -m mkb.migrate_to_v2 --dry-run               # preview only

No destructive operations. The v1 ``records`` table is left untouched so
existing readers (SQLiteStore.get_record, etc.) keep functioning. New code
should read ``facts`` going forward; a view ``facts_unified`` combines both
while we transition.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from app.config import DB_PATH


SCHEMA_FILE = Path(__file__).parent / "schema_v2.sql"
MIGRATION_ID = "schema_v2__2024_01"


_MIGRATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS migrations (
    id          TEXT PRIMARY KEY,
    applied_at  TEXT NOT NULL,
    notes       TEXT
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def already_applied(conn: sqlite3.Connection, migration_id: str) -> bool:
    conn.execute(_MIGRATIONS_TABLE)
    row = conn.execute(
        "SELECT 1 FROM migrations WHERE id=?", (migration_id,)
    ).fetchone()
    return row is not None


def apply_schema(conn: sqlite3.Connection, schema_sql: str) -> None:
    conn.executescript(schema_sql)


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def backfill_records_to_facts(conn: sqlite3.Connection, dry_run: bool = False) -> int:
    """Copy rows from v1 ``records`` into v2 ``facts`` (once)."""
    if not table_exists(conn, "records"):
        logger.info("No v1 'records' table found; nothing to back-fill.")
        return 0

    # Skip rows we've already imported.
    existing_ids = {
        r[0] for r in conn.execute("SELECT id FROM facts").fetchall()
    }

    migrated = 0
    rows = conn.execute("SELECT * FROM records").fetchall()
    now = datetime.utcnow().isoformat()

    for row in rows:
        if row["id"] in existing_ids:
            continue

        structured = _safe_json(row["structured_json"]) or {}
        raw_value = structured.get("value")
        raw_unit = structured.get("unit")
        raw_date = structured.get("date")

        payload = {
            "id":                    row["id"],
            "fact_type":             row["fact_type"],
            "entity_type":           row["fact_type"],
            "entity_name":           _extract_name(row, structured),
            "canonical_name":        None,
            "value":                 str(raw_value) if raw_value is not None else None,
            "unit":                  raw_unit,
            "normalized_value":      _as_float(raw_value),
            "normalized_unit":       raw_unit,
            "date":                  raw_date,
            "date_range_start":      None,
            "date_range_end":        None,
            "temporal_confidence":   1.0 if raw_date else 0.5,
            "is_negated":            0,
            "negation_type":         None,
            "subject":               "patient",
            "certainty":             "confirmed",
            "ocr_confidence":        None,
            "ocr_errors_detected":   0,
            "trust_level":           row["trust_level"],
            "confidence":            row["confidence"],
            "tier":                  row["tier"],
            "status":                row["status"],
            "occurrence_count":      1,
            "first_recorded":        row["first_recorded"],
            "last_confirmed":        row["last_confirmed"],
            "specialty":             row["specialty"],
            "source_type":           row["source_type"],
            "source_name":           row["source_name"] or "",
            "source_url":            row["source_url"],
            "source_document_id":    None,
            "extraction_method":     row["extraction_method"] or "claude",
            "raw_text":              row["content"],
            "parent_fact_id":        None,
            "superseded_by":         None,
            "requires_review":       row["requires_review"],
            "conflict_id":           None,
            "session_id":            row["session_id"] or "",
            "metadata_json":         json.dumps(structured),
        }

        if not dry_run:
            conn.execute(
                """
                INSERT INTO facts (
                    id, fact_type, entity_type, entity_name, canonical_name,
                    value, unit, normalized_value, normalized_unit,
                    date, date_range_start, date_range_end, temporal_confidence,
                    is_negated, negation_type, subject, certainty,
                    ocr_confidence, ocr_errors_detected,
                    trust_level, confidence, tier, status,
                    occurrence_count, first_recorded, last_confirmed,
                    specialty, source_type, source_name, source_url, source_document_id,
                    extraction_method, raw_text, parent_fact_id, superseded_by,
                    requires_review, conflict_id, session_id, metadata_json
                ) VALUES (
                    :id, :fact_type, :entity_type, :entity_name, :canonical_name,
                    :value, :unit, :normalized_value, :normalized_unit,
                    :date, :date_range_start, :date_range_end, :temporal_confidence,
                    :is_negated, :negation_type, :subject, :certainty,
                    :ocr_confidence, :ocr_errors_detected,
                    :trust_level, :confidence, :tier, :status,
                    :occurrence_count, :first_recorded, :last_confirmed,
                    :specialty, :source_type, :source_name, :source_url, :source_document_id,
                    :extraction_method, :raw_text, :parent_fact_id, :superseded_by,
                    :requires_review, :conflict_id, :session_id, :metadata_json
                )
                """,
                payload,
            )
            conn.execute(
                """
                INSERT INTO provenance (
                    fact_id, source_type, source_name, source_url,
                    raw_text, extracted_at, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["id"], row["source_type"], row["source_name"] or "",
                    row["source_url"], row["content"], now, row["confidence"],
                ),
            )
        migrated += 1

    logger.info(f"Back-filled {migrated} fact(s) from 'records' into 'facts'.")
    return migrated


def create_unified_view(conn: sqlite3.Connection) -> None:
    """Create a read-only view that unions v1 and v2 fact rows."""
    conn.execute("DROP VIEW IF EXISTS facts_unified")
    if table_exists(conn, "records") and table_exists(conn, "facts"):
        conn.execute(
            """
            CREATE VIEW facts_unified AS
            SELECT id, fact_type, entity_type, entity_name, canonical_name,
                   value, unit, date, tier, status, trust_level, confidence,
                   specialty, source_type, source_name, first_recorded
              FROM facts
            UNION ALL
            SELECT id, fact_type, fact_type AS entity_type, content AS entity_name,
                   NULL AS canonical_name, NULL AS value, NULL AS unit, NULL AS date,
                   tier, status, trust_level, confidence, specialty,
                   source_type, source_name, first_recorded
              FROM records
             WHERE id NOT IN (SELECT id FROM facts)
            """
        )


def record_migration(conn: sqlite3.Connection, migration_id: str, notes: str = "") -> None:
    conn.execute(
        "INSERT OR IGNORE INTO migrations (id, applied_at, notes) VALUES (?, ?, ?)",
        (migration_id, datetime.utcnow().isoformat(), notes),
    )


def migrate(db_path: Path = DB_PATH, dry_run: bool = False) -> dict[str, Any]:
    """Run the migration. Returns a summary dict."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if not SCHEMA_FILE.exists():
        raise FileNotFoundError(f"Schema file not found: {SCHEMA_FILE}")

    schema_sql = SCHEMA_FILE.read_text(encoding="utf-8")

    with connect(db_path) as conn:
        first_run = not already_applied(conn, MIGRATION_ID)

        if dry_run:
            logger.info(f"[dry-run] Would apply {MIGRATION_ID} to {db_path} "
                        f"(first_run={first_run})")
            rows = conn.execute("SELECT COUNT(*) FROM records").fetchone()[0] \
                if table_exists(conn, "records") else 0
            return {
                "db_path": str(db_path),
                "migration_id": MIGRATION_ID,
                "first_run": first_run,
                "dry_run": True,
                "records_to_migrate": rows,
                "schema_applied": False,
            }

        apply_schema(conn, schema_sql)
        migrated = backfill_records_to_facts(conn, dry_run=False)
        create_unified_view(conn)
        record_migration(conn, MIGRATION_ID,
                         notes=f"backfilled={migrated}")
        conn.commit()

    logger.info(f"Migration {MIGRATION_ID} applied to {db_path}")
    return {
        "db_path": str(db_path),
        "migration_id": MIGRATION_ID,
        "first_run": first_run,
        "dry_run": False,
        "records_migrated": migrated,
        "schema_applied": True,
    }


# ── helpers ──────────────────────────────────────────────────────────────────

def _safe_json(raw: Any) -> dict:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}


def _extract_name(row: sqlite3.Row, structured: dict) -> str:
    for key in ("entity_name", "name", "test_name", "drug", "title"):
        if structured.get(key):
            return str(structured[key])
    return (row["content"] or "")[:200]


def _as_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply MedAI MKB schema v2.")
    parser.add_argument("--db-path", type=Path, default=DB_PATH,
                        help="Path to SQLite database (defaults to config.DB_PATH).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Do not modify the database; print what would change.")
    args = parser.parse_args(argv)

    summary = migrate(db_path=args.db_path, dry_run=args.dry_run)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
