from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from app.startup_preflight import (
    build_startup_diagnostics,
    initialize_startup_state,
    safe_relative_path,
)


def test_sqlite_init_success_path(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "mkb.db"
    db_path.parent.mkdir(parents=True)
    sqlite3.connect(db_path).close()

    state = initialize_startup_state(lambda: {"sql": object()})

    assert state.ok is True
    assert state.components is not None


def test_memory_error_during_init_returns_degraded_state() -> None:
    def fail():
        raise MemoryError("synthetic startup memory failure")

    state = initialize_startup_state(fail)

    assert state.ok is False
    assert state.components is None
    assert state.diagnostics.exception_class == "MemoryError"
    assert state.diagnostics.exception_category == "sqlite_init_memory_error"


def test_corrupt_invalid_db_path_handled_safely(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "mkb.db"
    db_path.parent.mkdir(parents=True)
    db_path.write_text("not sqlite content", encoding="utf-8")

    diagnostics = build_startup_diagnostics(db_path=db_path, exception=sqlite3.DatabaseError("bad db"))

    assert diagnostics.db_file_exists is True
    assert diagnostics.sqlite_connect_result == "connect_failed_database_error"
    assert diagnostics.sqlite_quick_check_result == "quick_check_not_available"
    assert diagnostics.exception_category == "sqlite_database_error"


def test_missing_db_parent_directory_handled_safely(tmp_path: Path) -> None:
    db_path = tmp_path / "missing" / "mkb.db"

    diagnostics = build_startup_diagnostics(db_path=db_path)

    assert diagnostics.db_parent_exists is False
    assert diagnostics.db_file_exists is False
    assert diagnostics.db_file_size_bucket == "missing"
    assert diagnostics.sqlite_connect_result == "not_run_missing_db"


def test_ui_import_does_not_crash_if_mkb_init_fails() -> None:
    import app.main as main

    assert hasattr(main, "render_degraded_startup_panel")
    assert hasattr(main, "main")


def test_no_private_absolute_path_leaks_in_diagnostic_summary(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "mkb.db"
    diagnostics = build_startup_diagnostics(db_path=db_path, exception=MemoryError())
    payload = json.dumps(diagnostics.safe_public_summary())

    assert str(tmp_path) not in payload
    assert "\\" not in diagnostics.db_path_label


def test_no_db_contents_printed(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "mkb.db"
    db_path.parent.mkdir(parents=True)
    db_path.write_text("PRIVATE_DB_CONTENT_SHOULD_NOT_PRINT", encoding="utf-8")

    diagnostics = build_startup_diagnostics(db_path=db_path, exception=sqlite3.DatabaseError("bad db"))
    payload = json.dumps(diagnostics.safe_public_summary())

    assert "PRIVATE_DB_CONTENT_SHOULD_NOT_PRINT" not in payload


def test_no_clinical_processing_when_mkb_unavailable() -> None:
    state = initialize_startup_state(lambda: (_ for _ in ()).throw(MemoryError("fail")))

    assert state.ok is False
    assert state.components is None


def test_operator_control_panel_import_still_works() -> None:
    from app.operator_control_panel import COMMAND_ALLOWLIST

    assert "git_safety_check" in COMMAND_ALLOWLIST


def test_start_launchers_document_local_only_assumptions() -> None:
    normal = Path("Start_MedAI_UI.bat").read_text(encoding="utf-8")
    test = Path("Start_MedAI_Test_UI.bat").read_text(encoding="utf-8")

    for text in (normal, test):
        assert "MEDAI_LOCAL_ONLY=1" in text
        assert "MEDAI_ALLOW_EXTERNAL_API=0" in text


def test_safe_relative_path_uses_repo_relative_label() -> None:
    repo_file = Path("data/mkb.db")

    assert safe_relative_path(repo_file) == "data/mkb.db"
