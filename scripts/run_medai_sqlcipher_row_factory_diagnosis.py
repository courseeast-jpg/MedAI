"""Diagnose SQLCipher/sqlite3 row factory compatibility without row dumps."""
from __future__ import annotations

import inspect
import json
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import DB_PATH  # noqa: E402
from app.startup_preflight import safe_relative_path  # noqa: E402
from clinical_knowledge.privacy.report_privacy import check_public_report_payload  # noqa: E402
from mkb import sqlite_store  # noqa: E402
from mkb.sqlite_store import ENCRYPTED, SQLiteStore, dict_row_factory, first_column_value, sqlite_backend  # noqa: E402


REPORT_DIR = REPO_ROOT / "reports" / "medai_db_repair_02"


def _safe_exception(exc: BaseException) -> dict[str, str]:
    return {"exception_class": exc.__class__.__name__, "exception_category": "row_factory_or_backend_error"}


class FakeSqlcipherCursor:
    description = (("id", None, None, None, None, None, None),)


def simulate_sqlite3_row_incompatibility() -> dict[str, Any]:
    try:
        sqlite3.Row(FakeSqlcipherCursor(), ("value",))
        return {"sqlite3_row_rejects_fake_sqlcipher_cursor": False, "exception_class": None}
    except TypeError as exc:
        return {"sqlite3_row_rejects_fake_sqlcipher_cursor": True, "exception_class": exc.__class__.__name__}


def inspect_row_factory_source() -> dict[str, Any]:
    source = inspect.getsource(sqlite_store.SQLiteStore._get_conn)
    return {
        "sqlite3_row_direct_assignment_present": "sqlite3.Row" in source,
        "dict_row_factory_present": "dict_row_factory" in source,
    }


def run_temp_store_count() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "mkb.db"
        try:
            store = SQLiteStore(db_path, "synthetic_validation_key" if ENCRYPTED else "")
            counts = store.count_records()
            return {
                "result": "ok",
                "counts_keys": sorted(counts.keys()),
                "counts_total": counts["total"],
                "backend_module": sqlite_backend.__name__,
            }
        except BaseException as exc:
            payload = {"result": "failed", "backend_module": sqlite_backend.__name__}
            payload.update(_safe_exception(exc))
            return payload


def run_active_store_count() -> dict[str, Any]:
    if not DB_PATH.exists():
        return {"result": "skipped_missing_db", "db_label": safe_relative_path(DB_PATH)}
    try:
        store = SQLiteStore(DB_PATH, "default_dev_key" if ENCRYPTED else "")
        counts = store.count_records()
        return {
            "result": "ok",
            "db_label": safe_relative_path(DB_PATH),
            "counts_keys": sorted(counts.keys()),
            "counts_total_bucket": "zero" if counts["total"] == 0 else "nonzero",
        }
    except BaseException as exc:
        payload = {"result": "failed", "db_label": safe_relative_path(DB_PATH)}
        payload.update(_safe_exception(exc))
        return payload


def validate_backend_neutral_helpers() -> dict[str, Any]:
    class Cursor:
        description = (("count",),)

    row = dict_row_factory(Cursor(), (3,))
    return {
        "dict_row_factory_result_type": row.__class__.__name__,
        "first_column_value": first_column_value(row),
        "passed": row == {"count": 3} and first_column_value(row) == 3,
    }


def diagnose() -> dict[str, Any]:
    source = inspect_row_factory_source()
    active_count = run_active_store_count()
    temp_count = run_temp_store_count()
    helper_check = validate_backend_neutral_helpers()
    simulation = simulate_sqlite3_row_incompatibility()
    if source["sqlite3_row_direct_assignment_present"] and ENCRYPTED:
        category = "sqlite3_row_used_with_sqlcipher_backend"
    elif temp_count["result"] != "ok":
        category = "count_records_backend_incompatible"
    elif helper_check["passed"] and source["dict_row_factory_present"]:
        category = "row_factory_backend_compatible"
    else:
        category = "unknown_needs_manual_review"
    return {
        "block_id": "MEDAI-DB-REPAIR-02",
        "conclusion": "medai_sqlcipher_row_factory_diagnosis_ready",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnosis_category": category,
        "backend_module": sqlite_backend.__name__,
        "sqlcipher_available": ENCRYPTED,
        "source_inspection": source,
        "sqlite3_row_incompatibility_simulation": simulation,
        "temp_store_count_records": temp_count,
        "active_store_count_records": active_count,
        "active_store_open_required_for_row_factory_diagnosis": False,
        "backend_neutral_helper_check": helper_check,
        "db_rows_printed": False,
        "db_key_printed": False,
        "external_api_used": False,
        "import_performed": False,
    }


def write_reports(report: dict[str, Any], report_dir: Path = REPORT_DIR) -> tuple[Path, Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "medai_db_repair_02_report.json"
    md_path = report_dir / "medai_db_repair_02_report.md"
    guide_path = report_dir / "MEDAI_DB_REPAIR_02.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# MEDAI-DB-REPAIR-02 Report",
                "",
                f"Conclusion: `{report['conclusion']}`",
                f"Diagnosis category: `{report['diagnosis_category']}`",
                f"Backend module: `{report.get('backend_module', report.get('diagnosis', {}).get('backend_module', 'unknown'))}`",
                f"Active local DB count result: `{report.get('active_local_db_count_records_result', report.get('active_store_count_records', {}).get('result', 'not_recorded'))}`",
                f"Real DB modified: `{report.get('real_db_modified', False)}`",
                "",
                "## Safety",
                "",
                "- DB rows printed: false",
                "- DB key printed: false",
                "- External API used: false",
                "- Import performed: false",
                "",
                "## Next Step",
                "",
                report.get("next_manual_launch_step", "Review diagnostics before launch."),
                "",
            ]
        ),
        encoding="utf-8",
    )
    guide_path.write_text(
        "\n".join(
            [
                "# MEDAI-DB-REPAIR-02 SQLCipher Row Factory Compatibility",
                "",
                "This block validates that MKB row handling works with sqlite3 and SQLCipher backends.",
                "",
                "The fix uses a backend-neutral dictionary row factory and keeps aggregate count queries row-type agnostic.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return json_path, md_path, guide_path


def main() -> int:
    report = diagnose()
    paths = write_reports(report)
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
                "diagnosis_category": report["diagnosis_category"],
                "report_json": safe_relative_path(paths[0]),
            },
            indent=2,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
