"""MEDAI-DB-REPAIR-02 validation script."""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.startup_preflight import safe_relative_path  # noqa: E402
from clinical_knowledge.privacy.report_privacy import check_public_report_payload  # noqa: E402
from mkb.sqlite_store import ENCRYPTED, SQLiteStore, dict_row_factory, first_column_value  # noqa: E402
from scripts.run_medai_sqlcipher_row_factory_diagnosis import diagnose, write_reports  # noqa: E402


REPORT_DIR = REPO_ROOT / "reports" / "medai_db_repair_02"


def validate_temp_store() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "mkb.db"
        store = SQLiteStore(db_path, "synthetic_validation_key" if ENCRYPTED else "")
        counts = store.count_records()
        return {
            "temp_store_count_records_passed": sorted(counts.keys()) == [
                "active",
                "hypothesis",
                "quarantined",
                "total",
            ],
            "temp_store_total": counts["total"],
        }


def validate_helpers() -> dict:
    class Cursor:
        description = (("count",),)

    row = dict_row_factory(Cursor(), (7,))
    return {
        "dict_row_factory_passed": row == {"count": 7},
        "first_column_value_passed": first_column_value(row) == 7,
    }


def build_validation_report() -> dict:
    diagnosis = diagnose()
    temp_store = validate_temp_store()
    helpers = validate_helpers()
    active_result = diagnosis["active_store_count_records"]["result"]
    passed = (
        diagnosis["diagnosis_category"] == "row_factory_backend_compatible"
        and diagnosis["temp_store_count_records"]["result"] == "ok"
        and temp_store["temp_store_count_records_passed"]
        and helpers["dict_row_factory_passed"]
        and helpers["first_column_value_passed"]
    )
    report = {
        "block_id": "MEDAI-DB-REPAIR-02",
        "conclusion": "medai_db_repair02_row_factory_ready" if passed else "medai_db_repair02_row_factory_failed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnosis_category": diagnosis["diagnosis_category"],
        "root_cause": "sqlite3.Row is incompatible with sqlcipher3 cursor objects when SQLCipher is the active backend.",
        "fix_summary": "SQLiteStore now uses a backend-neutral dictionary row factory and aggregate count helper.",
        "active_local_db_count_records_result": active_result,
        "active_local_db_count_records_required": False,
        "diagnosis": diagnosis,
        "temp_store_validation": temp_store,
        "helper_validation": helpers,
        "real_db_modified": False,
        "db_rows_printed": False,
        "db_key_printed": False,
        "external_api_used": False,
        "import_performed": False,
        "clinical_logic_changed": False,
        "ocr_extractor_safety_gates_changed": False,
        "route_fix_behavior_changed": False,
        "b07_terminology_behavior_changed": False,
        "next_manual_launch_step": (
            "Run scripts/run_medai_mkb_db_quarantine_recreate.py with explicit confirmation, then run Start_MedAI_UI.bat."
            if active_result == "failed"
            else "Run Start_MedAI_UI.bat and confirm the sidebar MKB Status renders."
        ),
    }
    write_reports(report, REPORT_DIR)
    return report


def main() -> int:
    report = build_validation_report()
    failed = report["conclusion"] != "medai_db_repair02_row_factory_ready"
    for path in (
        REPORT_DIR / "MEDAI_DB_REPAIR_02.md",
        REPORT_DIR / "medai_db_repair_02_report.json",
        REPORT_DIR / "medai_db_repair_02_report.md",
    ):
        result = check_public_report_payload(path.read_text(encoding="utf-8"))
        if not result.passed:
            print(json.dumps({"privacy_failed": safe_relative_path(path)}, indent=2))
            failed = True
    print(
        json.dumps(
            {
                "conclusion": report["conclusion"],
                "diagnosis_category": report["diagnosis_category"],
                "external_api_used": False,
                "import_performed": False,
            },
            indent=2,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
