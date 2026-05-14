from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from app.startup_preflight import build_startup_diagnostics
from scripts.run_medai_mkb_db_diagnostics import build_report


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_missing_db_classified_safely(tmp_path: Path) -> None:
    report = build_report(tmp_path / "missing" / "mkb.db")

    assert report["db_category"] == "missing"
    assert report["db_file_exists"] is False
    assert report["db_modified"] is False


def test_empty_placeholder_db_classified_safely(tmp_path: Path) -> None:
    db_path = tmp_path / "mkb.db"
    db_path.touch()

    report = build_report(db_path)

    assert report["db_category"] == "placeholder_or_empty"
    assert report["db_file_size_bucket"] == "empty"


def test_invalid_db_classified_without_content_leak(tmp_path: Path) -> None:
    db_path = tmp_path / "mkb.db"
    db_path.write_text("PRIVATE_DB_CONTENT_SHOULD_NOT_PRINT", encoding="utf-8")

    report = build_report(db_path)
    payload = json.dumps(report)

    assert report["db_category"] in {"encrypted_or_wrong_key", "corrupt_or_unreadable"}
    assert "PRIVATE_DB_CONTENT_SHOULD_NOT_PRINT" not in payload


def test_valid_plain_sqlite_classified_safely(tmp_path: Path) -> None:
    db_path = tmp_path / "mkb.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE records (id TEXT)")
        conn.execute("INSERT INTO records VALUES ('ROW_VALUE_SHOULD_NOT_PRINT')")

    report = build_report(db_path)
    payload = json.dumps(report)

    assert report["db_header_category"] == "sqlite_header"
    assert report["plain_sqlite_probe"]["quick_check"] == "ok"
    assert report["db_category"] in {
        "valid_plain_sqlite",
        "valid_plain_sqlite_but_sqlcipher_keyed_init_mismatch",
    }
    assert "ROW_VALUE_SHOULD_NOT_PRINT" not in payload


def test_diagnostics_use_relative_label_for_repo_db() -> None:
    diagnostics = build_startup_diagnostics(db_path=Path("data/mkb.db"), exception=MemoryError())
    payload = json.dumps(diagnostics.safe_public_summary())

    assert diagnostics.db_path_label == "data/mkb.db"
    assert ":\\" not in payload
    assert diagnostics.db_startup_category in {
        "missing",
        "placeholder_or_empty",
        "valid_plain_sqlite_but_sqlcipher_keyed_init_failed",
        "sqlite_init_memory_error",
        "valid_readonly_metadata",
        "unknown_needs_operator_decision",
    }


def test_quarantine_recreate_requires_explicit_confirmation(tmp_path: Path) -> None:
    db_path = tmp_path / "mkb.db"
    db_path.write_bytes(b"not a db")
    backup_root = tmp_path / "quarantine"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_medai_mkb_db_quarantine_recreate.py",
            "--db-path",
            str(db_path),
            "--backup-root",
            str(backup_root),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert db_path.exists()
    assert not backup_root.exists()
    assert "blocked_confirmation_required" in result.stdout


def test_quarantine_recreate_confirmed_works_on_temp_db(tmp_path: Path) -> None:
    db_path = tmp_path / "mkb.db"
    db_path.write_bytes(b"not a db")
    backup_root = tmp_path / "quarantine"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_medai_mkb_db_quarantine_recreate.py",
            "--db-path",
            str(db_path),
            "--backup-root",
            str(backup_root),
            "--confirm-quarantine-recreate",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["db_modified"] is True
    assert payload["backup_or_quarantine_created"] is True
    assert db_path.exists()
    assert list(backup_root.glob("mkb_quarantine_*.db"))


def test_default_backup_root_is_under_ignored_data_folder() -> None:
    script = Path("scripts/run_medai_mkb_db_quarantine_recreate.py").read_text(encoding="utf-8")
    gitignore = Path(".gitignore").read_text(encoding="utf-8")

    assert 'default="data/mkb_quarantine"' in script
    assert "data/" in gitignore


def test_diagnostics_script_writes_privacy_safe_reports(tmp_path: Path) -> None:
    db_path = tmp_path / "mkb.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE records (id TEXT)")
        conn.execute("INSERT INTO records VALUES ('PRIVATE_ROW_SHOULD_NOT_PRINT')")
    report_dir = tmp_path / "reports"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_medai_mkb_db_diagnostics.py",
            "--db-path",
            str(db_path),
            "--report-dir",
            str(report_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    combined = result.stdout + (report_dir / "medai_db_repair_01_report.json").read_text(encoding="utf-8")
    assert "PRIVATE_ROW_SHOULD_NOT_PRINT" not in combined
    assert str(tmp_path) not in combined
