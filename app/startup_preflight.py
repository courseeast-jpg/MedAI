"""Safe startup diagnostics for MedAI UI initialization.

This module performs metadata-only checks and wraps startup initialization so
the Streamlit UI can render a degraded diagnostics panel if the MKB store fails
to initialize.
"""
from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.config import DB_PATH
from mkb.sqlite_store import DB_SCHEMA


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class StartupDiagnostics:
    db_path_label: str
    db_parent_exists: bool
    db_file_exists: bool
    db_file_size_bucket: str
    db_header_category: str
    db_schema_length: int
    db_schema_sha256_12: str
    sqlite_connect_result: str
    sqlite_quick_check_result: str
    db_startup_category: str
    exception_class: str | None = None
    exception_category: str | None = None
    safe_operator_guidance: tuple[str, ...] = ()

    def safe_public_summary(self) -> dict[str, Any]:
        return {
            "db_path_label": self.db_path_label,
            "db_parent_exists": self.db_parent_exists,
            "db_file_exists": self.db_file_exists,
            "db_file_size_bucket": self.db_file_size_bucket,
            "db_header_category": self.db_header_category,
            "db_schema_length": self.db_schema_length,
            "db_schema_sha256_12": self.db_schema_sha256_12,
            "sqlite_connect_result": self.sqlite_connect_result,
            "sqlite_quick_check_result": self.sqlite_quick_check_result,
            "db_startup_category": self.db_startup_category,
            "exception_class": self.exception_class,
            "exception_category": self.exception_category,
            "safe_operator_guidance": list(self.safe_operator_guidance),
        }


@dataclass(frozen=True)
class StartupState:
    ok: bool
    components: dict[str, Any] | None
    diagnostics: StartupDiagnostics


def safe_relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def file_size_bucket(path: Path) -> str:
    if not path.exists():
        return "missing"
    size = path.stat().st_size
    if size == 0:
        return "empty"
    if size < 1024 * 1024:
        return "small_under_1mb"
    if size < 100 * 1024 * 1024:
        return "medium_under_100mb"
    if size < 1024 * 1024 * 1024:
        return "large_under_1gb"
    return "huge_over_1gb"


def db_header_category(path: Path) -> str:
    if not path.exists():
        return "missing"
    if path.stat().st_size == 0:
        return "empty"
    try:
        with path.open("rb") as handle:
            header = handle.read(16)
    except OSError:
        return "unreadable_header"
    if header == b"SQLite format 3\x00":
        return "sqlite_header"
    return "encrypted_or_unknown"


def categorize_exception(exc: BaseException | None) -> str | None:
    if exc is None:
        return None
    if isinstance(exc, MemoryError):
        return "sqlite_init_memory_error"
    if isinstance(exc, sqlite3.DatabaseError):
        return "sqlite_database_error"
    if isinstance(exc, OSError):
        return "filesystem_or_path_error"
    return "unexpected_startup_error"


def schema_summary(schema_text: str = DB_SCHEMA) -> tuple[int, str]:
    digest = hashlib.sha256(schema_text.encode("utf-8", errors="replace")).hexdigest()[:12]
    return len(schema_text), digest


def sqlite_metadata_probe(db_path: Path) -> tuple[str, str]:
    if not db_path.exists():
        return "not_run_missing_db", "not_run_missing_db"
    try:
        uri = f"file:{db_path.as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=2) as conn:
            quick = conn.execute("PRAGMA quick_check").fetchone()
        quick_value = str(quick[0]) if quick else "no_result"
        return "connect_ok_read_only", "ok" if quick_value.lower() == "ok" else "quick_check_non_ok"
    except sqlite3.DatabaseError:
        return "connect_failed_database_error", "quick_check_not_available"
    except OSError:
        return "connect_failed_filesystem_error", "quick_check_not_available"
    except Exception:
        return "connect_failed_unexpected_error", "quick_check_not_available"


def categorize_db_startup_state(
    *,
    header_category: str,
    sqlite_connect_result: str,
    sqlite_quick_check_result: str,
    exception_category: str | None,
) -> str:
    if header_category == "missing":
        return "missing"
    if header_category == "empty":
        return "placeholder_or_empty"
    if (
        header_category == "sqlite_header"
        and sqlite_connect_result == "connect_ok_read_only"
        and sqlite_quick_check_result == "ok"
        and exception_category == "sqlite_init_memory_error"
    ):
        return "valid_plain_sqlite_but_sqlcipher_keyed_init_failed"
    if exception_category == "sqlite_database_error":
        return "sqlite_init_database_error"
    if exception_category == "sqlite_init_memory_error":
        return "sqlite_init_memory_error"
    if sqlite_connect_result != "connect_ok_read_only" and header_category == "encrypted_or_unknown":
        return "encrypted_or_wrong_key_or_unreadable"
    if sqlite_connect_result == "connect_ok_read_only" and sqlite_quick_check_result == "ok":
        return "valid_readonly_metadata"
    return "unknown_needs_operator_decision"


def build_startup_diagnostics(
    *,
    db_path: Path = DB_PATH,
    schema_text: str = DB_SCHEMA,
    exception: BaseException | None = None,
) -> StartupDiagnostics:
    schema_len, schema_hash = schema_summary(schema_text)
    connect_result, quick_result = sqlite_metadata_probe(db_path)
    header = db_header_category(db_path)
    category = categorize_exception(exception)
    db_startup_category = categorize_db_startup_state(
        header_category=header,
        sqlite_connect_result=connect_result,
        sqlite_quick_check_result=quick_result,
        exception_category=category,
    )
    guidance = (
        "MKB initialization failed.",
        "No clinical processing started.",
        "Run Git Safety Check or startup diagnostics.",
        "Avoid manual DB deletion.",
        "Use a Codex repair block if database quarantine or restore is needed.",
    )
    return StartupDiagnostics(
        db_path_label=safe_relative_path(db_path),
        db_parent_exists=db_path.parent.exists(),
        db_file_exists=db_path.exists(),
        db_file_size_bucket=file_size_bucket(db_path),
        db_header_category=header,
        db_schema_length=schema_len,
        db_schema_sha256_12=schema_hash,
        sqlite_connect_result=connect_result,
        sqlite_quick_check_result=quick_result,
        db_startup_category=db_startup_category,
        exception_class=exception.__class__.__name__ if exception else None,
        exception_category=category,
        safe_operator_guidance=guidance if exception else (),
    )


def initialize_startup_state(component_factory: Callable[[], dict[str, Any]]) -> StartupState:
    try:
        components = component_factory()
        return StartupState(
            ok=True,
            components=components,
            diagnostics=build_startup_diagnostics(),
        )
    except (MemoryError, sqlite3.DatabaseError, OSError, Exception) as exc:
        return StartupState(
            ok=False,
            components=None,
            diagnostics=build_startup_diagnostics(exception=exc),
        )
