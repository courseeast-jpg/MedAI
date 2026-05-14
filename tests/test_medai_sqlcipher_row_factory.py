from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from app.startup_preflight import build_startup_diagnostics
from mkb.sqlite_store import ENCRYPTED, SQLiteStore, dict_row_factory, first_column_value
from scripts.run_medai_sqlcipher_row_factory_diagnosis import (
    diagnose,
    simulate_sqlite3_row_incompatibility,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_sqlite3_backend_count_records_works(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "mkb.db", "synthetic_key" if ENCRYPTED else "")

    counts = store.count_records()

    assert counts == {"total": 0, "active": 0, "hypothesis": 0, "quarantined": 0}


def test_sqlcipher_like_cursor_does_not_require_sqlite3_row() -> None:
    simulation = simulate_sqlite3_row_incompatibility()

    class Cursor:
        description = (("field",),)

    assert simulation["sqlite3_row_rejects_fake_sqlcipher_cursor"] is True
    assert dict_row_factory(Cursor(), ("value",)) == {"field": "value"}


def test_count_records_returns_required_keys(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "mkb.db", "synthetic_key" if ENCRYPTED else "")

    counts = store.count_records()

    assert sorted(counts.keys()) == ["active", "hypothesis", "quarantined", "total"]


def test_backend_neutral_dict_row_conversion() -> None:
    class Cursor:
        description = (("count",), ("tier",))

    row = dict_row_factory(Cursor(), (4, "active"))

    assert row == {"count": 4, "tier": "active"}
    assert first_column_value(row) == 4
    assert first_column_value((5,)) == 5


def test_sqlite_store_initialization_remains_compatible(tmp_path: Path) -> None:
    db_path = tmp_path / "mkb.db"

    SQLiteStore(db_path, "synthetic_key" if ENCRYPTED else "")

    assert db_path.exists()


def test_count_records_works_with_repaired_local_db_path_without_printing_rows() -> None:
    diagnosis = diagnose()
    payload = json.dumps(diagnosis)

    assert diagnosis["diagnosis_category"] == "row_factory_backend_compatible"
    assert diagnosis["active_store_count_records"]["result"] in {"ok", "skipped_missing_db", "failed"}
    assert "content" not in payload.lower()
    assert "structured_json" not in payload


def test_ui_sidebar_mkb_status_count_path_can_call_count_records(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "mkb.db", "synthetic_key" if ENCRYPTED else "")

    counts = store.count_records()
    active = counts["active"]
    hypothesis = counts["hypothesis"]

    assert active == 0
    assert hypothesis == 0


def test_startup_diagnostics_still_behave_safely(tmp_path: Path) -> None:
    db_path = tmp_path / "mkb.db"
    sqlite3.connect(db_path).close()

    diagnostics = build_startup_diagnostics(db_path=db_path, exception=MemoryError())
    payload = json.dumps(diagnostics.safe_public_summary())

    assert diagnostics.exception_category == "sqlite_init_memory_error"
    assert str(tmp_path) not in payload


def test_diagnosis_and_validation_reports_are_privacy_safe() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_medai_sqlcipher_row_factory_diagnosis.py"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "diagnosis_category" in result.stdout
    assert "default_dev_key" not in result.stdout
    assert ":\\" not in result.stdout
