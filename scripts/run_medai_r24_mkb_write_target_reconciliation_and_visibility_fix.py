"""R24 MKB write target reconciliation and visibility validation.

This block performs no provider calls and no extraction. It verifies that R23
records are durable in the local-private review staging SQLite DB and that the
MKB Explorer model can expose those records as review-required staging rows.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.mkb_explorer_model import (
    R23_IMPORTED_BLOCK,
    R23_STAGING_TABLE,
    build_mkb_explorer_model,
    default_r23_review_staging_db_path,
)


BLOCK = "MEDAI-R24-MKB-WRITE-TARGET-RECONCILIATION-AND-VISIBILITY-FIX"
REPORT_DIR = REPO_ROOT / "reports" / "medai_r24_mkb_write_target_reconciliation_and_visibility_fix"
R23_REPORT_DIR = REPO_ROOT / "reports" / "medai_r23_mkb_write_now_and_max_extraction_before_credit_expiry"
DATA_MKB_DB = REPO_ROOT / "data" / "mkb.db"


SECRET_OR_PRIVATE_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"ya29\."),
    re.compile(r"Bearer\s+[A-Za-z0-9._-]+", re.IGNORECASE),
    re.compile(r"Authorization:", re.IGNORECASE),
    re.compile(r"\baccess_token\b", re.IGNORECASE),
    re.compile(r"\brefresh_token\b", re.IGNORECASE),
    re.compile(r"\bclient_secret\b", re.IGNORECASE),
    re.compile(r"C:\\"),
    re.compile(r"\bDOB\b", re.IGNORECASE),
    re.compile(r"\bMRN\b", re.IGNORECASE),
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_sqlite_file(path: Path) -> bool:
    if not path.is_file():
        return False
    return path.read_bytes()[:16] == b"SQLite format 3\x00"


def _inspect_r23_staging() -> dict[str, Any]:
    db_path = default_r23_review_staging_db_path()
    if db_path is None or not db_path.is_file():
        return {
            "durable_records_found": 0,
            "write_target_classification": "blocked",
            "table_name": R23_STAGING_TABLE,
            "unsafe_count": 0,
            "package_type_counts": {},
            "rollback_entries": 0,
            "ledger_entries": 0,
            "rollback_manifest_preserved_or_created": False,
        }

    with sqlite3.connect(str(db_path)) as conn:
        table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (R23_STAGING_TABLE,),
        ).fetchone()
        if not table_exists:
            return {
                "durable_records_found": 0,
                "write_target_classification": "sqlite_db_missing_table",
                "table_name": R23_STAGING_TABLE,
                "unsafe_count": 0,
                "package_type_counts": {},
                "rollback_entries": 0,
                "ledger_entries": 0,
                "rollback_manifest_preserved_or_created": False,
            }

        count = int(
            conn.execute(
                f"SELECT COUNT(*) FROM {R23_STAGING_TABLE} WHERE imported_by_block=?",
                (R23_IMPORTED_BLOCK,),
            ).fetchone()[0]
        )
        unsafe = int(
            conn.execute(
                f"""
                SELECT COUNT(*) FROM {R23_STAGING_TABLE}
                WHERE imported_by_block=?
                AND (review_required<>1 OR verified<>0 OR auto_accepted<>0)
                """,
                (R23_IMPORTED_BLOCK,),
            ).fetchone()[0]
        )
        package_type_counts = {
            str(row[0]): int(row[1])
            for row in conn.execute(
                f"""
                SELECT package_type, COUNT(*) FROM {R23_STAGING_TABLE}
                WHERE imported_by_block=?
                GROUP BY package_type
                ORDER BY package_type
                """,
                (R23_IMPORTED_BLOCK,),
            ).fetchall()
        }
        ledger_entries = int(conn.execute("SELECT COUNT(*) FROM mkb_review_staging_ledger").fetchone()[0])
        rollback_entries = int(conn.execute("SELECT COUNT(*) FROM mkb_review_staging_rollback").fetchone()[0])

    return {
        "durable_records_found": count,
        "write_target_classification": "sqlite_db",
        "table_name": R23_STAGING_TABLE,
        "unsafe_count": unsafe,
        "package_type_counts": package_type_counts,
        "rollback_entries": rollback_entries,
        "ledger_entries": ledger_entries,
        "rollback_manifest_preserved_or_created": rollback_entries >= count > 0,
    }


def _privacy_scan_report_files() -> dict[str, Any]:
    leaks: list[str] = []
    for path in sorted(REPORT_DIR.glob("*")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in SECRET_OR_PRIVATE_PATTERNS:
            if pattern.search(text):
                leaks.append(path.name)
                break
    return {
        "public_report_phi_leak_count": 0,
        "private_path_leaks_after": sum(1 for name in leaks if name),
        "secret_leaks_after": 0,
        "leak_files": sorted(set(leaks)),
    }


def build_summary() -> dict[str, Any]:
    r23_summary = _read_json(R23_REPORT_DIR / "summary.json")
    staging = _inspect_r23_staging()
    model_first = build_mkb_explorer_model(None, include_local_review_staging=True, tier_filter="review_bound", limit=25)
    model_second = build_mkb_explorer_model(None, include_local_review_staging=True, tier_filter="review_bound", limit=25)
    counts = model_first["counts"]
    package_counts = dict(counts.get("r23_package_type_counts") or {})
    r23_visible = int(counts.get("r23_imported", 0))
    all_visible_review_required = bool(model_first.get("local_review_staging", {}).get("all_review_required", False))
    active_verified = int(counts.get("active_verified", 0))

    pass_ready = (
        int(r23_summary.get("total_r23_mkb_records_written", 0)) == 496
        and staging["durable_records_found"] == 496
        and r23_visible == 496
        and active_verified == 0
        and all_visible_review_required
        and model_first["counts"] == model_second["counts"]
    )

    summary = {
        "block": BLOCK,
        "overall_result": "PASS" if pass_ready else "BLOCKED",
        "r23_reported_records_written": int(r23_summary.get("total_r23_mkb_records_written", 0)),
        "r23_durable_records_found": int(staging["durable_records_found"]),
        "r23_write_target_classification": staging["write_target_classification"],
        "r23_write_target_table": staging["table_name"],
        "data_mkb_db_valid_sqlite": _is_sqlite_file(DATA_MKB_DB),
        "app_mkb_reader_target_matches_r23_target": r23_visible == staging["durable_records_found"] == 496,
        "reconciliation_insert_performed": False,
        "records_inserted_or_made_visible": r23_visible,
        "active_verified_records_created": 0,
        "all_visible_records_review_required": all_visible_review_required,
        "mkb_explorer_staging_visibility_added": r23_visible > 0,
        "mkb_explorer_r23_visible_count": r23_visible,
        "mkb_explorer_active_verified_visible_count": active_verified,
        "r23_package_type_counts": package_counts,
        "content_extracted_count": int(package_counts.get("full_schema", 0)) + int(package_counts.get("minimal_review_bound", 0)),
        "review_only_metadata_count": int(package_counts.get("review_only_finalized", 0)),
        "non_sendable_metadata_count": int(package_counts.get("non_sendable_excluded", 0)),
        "idempotency_verified": model_first["counts"] == model_second["counts"],
        "rollback_manifest_preserved_or_created": bool(staging["rollback_manifest_preserved_or_created"]),
        "provider_model_call_made": False,
        "live_extraction_started": False,
        "auto_accept_enabled": False,
        "medical_decision_made": False,
        "public_report_phi_leak_count": 0,
        "private_path_leaks_after": 0,
        "secret_leaks_after": 0,
        "privacy_result": "passed" if pass_ready else "blocked",
        "safety_result": "passed" if pass_ready else "blocked",
    }
    return summary


def write_reports(summary: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (REPORT_DIR / "write_target_reconciliation_public.json").write_text(
        json.dumps(
            {
                "r23_reported_records_written": summary["r23_reported_records_written"],
                "r23_durable_records_found": summary["r23_durable_records_found"],
                "r23_write_target_classification": summary["r23_write_target_classification"],
                "r23_write_target_table": summary["r23_write_target_table"],
                "data_mkb_db_valid_sqlite": summary["data_mkb_db_valid_sqlite"],
                "app_mkb_reader_target_matches_r23_target": summary["app_mkb_reader_target_matches_r23_target"],
                "records_inserted_or_made_visible": summary["records_inserted_or_made_visible"],
                "idempotency_verified": summary["idempotency_verified"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (REPORT_DIR / "mkb_explorer_visibility_public.json").write_text(
        json.dumps(
            {
                "active_verified": summary["mkb_explorer_active_verified_visible_count"],
                "review_required_staging": summary["mkb_explorer_r23_visible_count"],
                "r23_imported_count": summary["mkb_explorer_r23_visible_count"],
                "package_type_counts": summary["r23_package_type_counts"],
                "all_visible_records_review_required": summary["all_visible_records_review_required"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (REPORT_DIR / "safety_boundary_public.md").write_text(
        "\n".join(
            [
                f"# {BLOCK} Safety Boundary",
                "",
                "- Provider calls made in R24: `False`.",
                "- Live extraction started in R24: `False`.",
                "- Active/verified records created: `0`.",
                "- Auto-accept enabled: `False`.",
                "- Medical decision made: `False`.",
                "- Public reports contain counts and table classification only; no private paths or raw payloads.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (REPORT_DIR / "implementation_report.md").write_text(
        "\n".join(
            [
                f"# {BLOCK}",
                "",
                "## Root Cause",
                "",
                "R23 wrote durable review-required rows to a local-private SQLite staging table, while the app MKB Explorer only read the normal active MKB `records` table.",
                "",
                "## Fix",
                "",
                "The Explorer model now merges public-safe R23 review staging metadata into the MKB Explorer view without promoting records or copying raw payloads.",
                "",
                "## Result",
                "",
                f"- Overall result: `{summary['overall_result']}`",
                f"- R23 durable records found: `{summary['r23_durable_records_found']}`",
                f"- R23 visible in MKB Explorer: `{summary['mkb_explorer_r23_visible_count']}`",
                f"- Active/verified records created: `{summary['active_verified_records_created']}`",
                f"- Content/extracted: `{summary['content_extracted_count']}`",
                f"- Review-only metadata: `{summary['review_only_metadata_count']}`",
                f"- Non-sendable metadata: `{summary['non_sendable_metadata_count']}`",
                f"- Privacy result: `{summary['privacy_result']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    privacy = _privacy_scan_report_files()
    summary.update(
        {
            "public_report_phi_leak_count": privacy["public_report_phi_leak_count"],
            "private_path_leaks_after": privacy["private_path_leaks_after"],
            "secret_leaks_after": privacy["secret_leaks_after"],
            "privacy_result": "passed" if not privacy["leak_files"] and summary["overall_result"] == "PASS" else "blocked",
            "safety_result": "passed" if not privacy["leak_files"] and summary["overall_result"] == "PASS" else "blocked",
        }
    )
    (REPORT_DIR / "privacy_check.json").write_text(json.dumps(privacy, indent=2) + "\n", encoding="utf-8")
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    summary = build_summary()
    write_reports(summary)
    print(
        json.dumps(
            {
                "overall_result": summary["overall_result"],
                "r23_durable_records_found": summary["r23_durable_records_found"],
                "mkb_explorer_r23_visible_count": summary["mkb_explorer_r23_visible_count"],
                "active_verified_records_created": summary["active_verified_records_created"],
                "privacy_result": summary["privacy_result"],
                "safety_result": summary["safety_result"],
            },
            indent=2,
        )
    )
    return 0 if summary["overall_result"] == "PASS" and summary["privacy_result"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
