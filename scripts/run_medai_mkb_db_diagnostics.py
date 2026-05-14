"""Safe diagnostics for the local MedAI MKB SQLite database.

The probe is metadata-only: no row dumps, no keys, no absolute local paths.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import DB_PATH  # noqa: E402
from app.startup_preflight import (  # noqa: E402
    db_header_category,
    file_size_bucket,
    safe_relative_path,
    schema_summary,
)
from clinical_knowledge.privacy.report_privacy import check_public_report_payload  # noqa: E402
from mkb.sqlite_store import DB_SCHEMA, ENCRYPTED, sqlite_backend  # noqa: E402


REPORT_DIR = REPO_ROOT / "reports" / "medai_db_repair_01"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run safe MKB DB diagnostics.")
    parser.add_argument("--db-path", default=str(DB_PATH), help="DB path relative to repo root.")
    parser.add_argument("--report-dir", default=str(REPORT_DIR), help="Report directory.")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _safe_exception_category(exc: BaseException) -> str:
    if isinstance(exc, MemoryError):
        return "memory_error"
    if isinstance(exc, sqlite3.DatabaseError):
        return "sqlite_database_error"
    if isinstance(exc, OSError):
        return "filesystem_error"
    return "unexpected_error"


def plain_sqlite_probe(db_path: Path) -> dict[str, Any]:
    if not db_path.exists():
        return {"connect": "not_run_missing", "quick_check": "not_run", "table_count": None}
    try:
        uri = f"file:{db_path.as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=2) as conn:
            quick = conn.execute("PRAGMA quick_check").fetchone()
            tables = conn.execute("SELECT count(*) FROM sqlite_master WHERE type='table'").fetchone()
        quick_value = str(quick[0]) if quick else "no_result"
        return {
            "connect": "ok",
            "quick_check": "ok" if quick_value.lower() == "ok" else "quick_check_non_ok",
            "table_count": int(tables[0]) if tables else 0,
        }
    except BaseException as exc:
        return {
            "connect": "failed",
            "quick_check": "not_available",
            "table_count": None,
            "exception_category": _safe_exception_category(exc),
            "exception_class": exc.__class__.__name__,
        }


def sqlcipher_probe(db_path: Path, *, key_value: str | None) -> dict[str, Any]:
    if not ENCRYPTED:
        return {"available": False, "attempted": False, "result": "sqlcipher_unavailable"}
    if not db_path.exists():
        return {"available": True, "attempted": False, "result": "not_run_missing"}
    if not key_value:
        return {"available": True, "attempted": False, "result": "not_run_no_key_configured"}
    try:
        with sqlite_backend.connect(str(db_path)) as conn:
            escaped_key = key_value.replace("'", "''")
            conn.execute(f"PRAGMA key = '{escaped_key}'")
            quick = conn.execute("PRAGMA quick_check").fetchone()
        quick_value = str(quick[0]) if quick else "no_result"
        return {
            "available": True,
            "attempted": True,
            "result": "ok" if quick_value.lower() == "ok" else "quick_check_non_ok",
        }
    except BaseException as exc:
        return {
            "available": True,
            "attempted": True,
            "result": "failed_with_configured_key",
            "exception_category": _safe_exception_category(exc),
            "exception_class": exc.__class__.__name__,
        }


def classify_db(
    *,
    exists: bool,
    header: str,
    plain: dict[str, Any],
    cipher: dict[str, Any],
    app_key_configured: bool,
) -> str:
    if not exists:
        return "missing"
    if header == "empty":
        return "placeholder_or_empty"
    if cipher.get("attempted") and cipher.get("result") == "ok":
        return "valid_sqlcipher_with_key"
    if plain.get("connect") == "ok" and plain.get("quick_check") == "ok":
        if ENCRYPTED and app_key_configured and cipher.get("result") == "failed_with_configured_key":
            return "valid_plain_sqlite_but_sqlcipher_keyed_init_mismatch"
        return "valid_plain_sqlite"
    if header == "encrypted_or_unknown" and cipher.get("result") == "failed_with_configured_key":
        return "encrypted_or_wrong_key"
    if plain.get("connect") != "ok":
        return "corrupt_or_unreadable"
    return "unknown_needs_manual_decision"


def build_report(db_path: Path) -> dict[str, Any]:
    app_key = os.getenv("DB_ENCRYPTION_KEY", "default_dev_key")
    app_key_configured = bool(app_key)
    schema_len, schema_hash = schema_summary(DB_SCHEMA)
    header = db_header_category(db_path)
    plain = plain_sqlite_probe(db_path)
    cipher = sqlcipher_probe(db_path, key_value=app_key if app_key_configured else None)
    category = classify_db(
        exists=db_path.exists(),
        header=header,
        plain=plain,
        cipher=cipher,
        app_key_configured=app_key_configured,
    )
    quarantine_recommended = category in {
        "valid_plain_sqlite_but_sqlcipher_keyed_init_mismatch",
        "corrupt_or_unreadable",
        "placeholder_or_empty",
        "encrypted_or_wrong_key",
    }
    return {
        "block_id": "MEDAI-DB-REPAIR-01",
        "conclusion": "medai_mkb_db_diagnostics_ready",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "db_path_label": safe_relative_path(db_path),
        "db_parent_exists": db_path.parent.exists(),
        "db_file_exists": db_path.exists(),
        "db_file_size_bucket": file_size_bucket(db_path),
        "db_header_category": header,
        "db_schema_length": schema_len,
        "db_schema_sha256_12": schema_hash,
        "sqlcipher_available": ENCRYPTED,
        "app_db_key_configured": app_key_configured,
        "db_key_value_printed": False,
        "plain_sqlite_probe": plain,
        "sqlcipher_probe": cipher,
        "db_category": category,
        "quarantine_recreate_available": True,
        "quarantine_recreate_recommended": quarantine_recommended,
        "db_modified": False,
        "backup_or_quarantine_created": False,
        "rows_printed": False,
        "external_api_used": False,
        "import_performed": False,
        "safe_next_action": (
            "Use the confirmation-gated quarantine/recreate command only after operator approval."
            if quarantine_recommended
            else "No DB repair action recommended from metadata diagnostics alone."
        ),
        "privacy_safety": {
            "raw_private_paths_in_report": False,
            "db_contents_printed": False,
            "phi_printed": False,
            "license_text_printed": False,
            "source_terminology_rows_printed": False,
        },
    }


def write_reports(report: dict[str, Any], report_dir: Path) -> tuple[Path, Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "medai_db_repair_01_report.json"
    md_path = report_dir / "medai_db_repair_01_report.md"
    guide_path = report_dir / "MEDAI_DB_REPAIR_01.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# MEDAI-DB-REPAIR-01 Report",
                "",
                f"Conclusion: `{report['conclusion']}`",
                "",
                f"- DB label: `{report['db_path_label']}`",
                f"- DB category: `{report['db_category']}`",
                f"- DB size bucket: `{report['db_file_size_bucket']}`",
                f"- Header category: `{report['db_header_category']}`",
                f"- SQLCipher available: `{report['sqlcipher_available']}`",
                f"- DB modified: `{report['db_modified']}`",
                f"- Backup or quarantine created: `{report['backup_or_quarantine_created']}`",
                "",
                "## Safe Next Action",
                "",
                report["safe_next_action"],
                "",
            ]
        ),
        encoding="utf-8",
    )
    guide_path.write_text(
        "\n".join(
            [
                "# MEDAI-DB-REPAIR-01 MKB DB Diagnostics",
                "",
                "This block diagnoses local MKB startup database state without printing rows or key values.",
                "",
                "The quarantine/recreate workflow is confirmation-gated and never runs automatically.",
                "",
                "Suggested operator sequence:",
                "",
                "1. Review the diagnostics report.",
                "2. Approve quarantine/recreate only when replacing the local startup DB with an empty schema DB is acceptable.",
                "3. Launch the UI after the approved repair step.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return json_path, md_path, guide_path


def main() -> int:
    args = parse_args()
    db_path = resolve_path(args.db_path)
    report_dir = resolve_path(args.report_dir)
    report = build_report(db_path)
    paths = write_reports(report, report_dir)
    failed = False
    for path in paths:
        result = check_public_report_payload(path.read_text(encoding="utf-8"))
        if not result.passed:
            print(json.dumps({"privacy_failed": safe_relative_path(path)}, indent=2))
            failed = True
    print(
        json.dumps(
            {
                "conclusion": report["conclusion"],
                "db_category": report["db_category"],
                "db_modified": False,
                "report_json": safe_relative_path(paths[0]),
            },
            indent=2,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
