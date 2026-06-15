"""Streamlit-free MKB Explorer model helpers.

MEDAI-UI-CAPABILITY-RESTORE-11B.

The model exposes public-safe record metadata for UI rendering and validation.
It reads only through the provided SQLite store and never emits raw source text,
raw filenames, private paths, or runtime DB rows.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

from app.config import DB_PATH
from app.schemas import MKBRecord
from app.specialty_selection import specialty_label, validate_specialty_key


R23_IMPORTED_BLOCK = "MEDAI-R23-MKB-WRITE-NOW-AND-MAX-EXTRACTION-BEFORE-CREDIT-EXPIRY"
R23_STAGING_TABLE = "mkb_review_staging_records"


def safe_record_id(record_id: str | None) -> str:
    value = str(record_id or "")
    if len(value) <= 12:
        return value
    return f"{value[:8]}...{value[-4:]}"


def display_content_public_safe(record: MKBRecord) -> str:
    content = str(record.content or "")
    structured = record.structured or {}
    fallback = (
        structured.get("test_name")
        or structured.get("text")
        or structured.get("name")
        or structured.get("description")
        or record.fact_type
    )
    if "[" in content and "]" in content:
        return str(fallback)
    return content[:160]


def _row_count(sql_store: Any, where: str = "", params: list[Any] | None = None) -> int:
    params = params or []
    with sql_store._get_conn() as conn:
        row = conn.execute(f"SELECT COUNT(*) AS c FROM records{where}", params).fetchone()
    return int(row["c"] if isinstance(row, dict) else row[0])


def default_r23_review_staging_db_path() -> Path | None:
    """Return the local-private R23 review staging DB path without exposing it in UI."""
    local_app_data = os.getenv("LOCALAPPDATA")
    if not local_app_data:
        return None
    return (
        Path(local_app_data)
        / "MedAI_Private"
        / "mkb"
        / "r23_write_now"
        / "medai_mkb_review_staging.sqlite3"
    )


def _uses_default_mkb_path(sql_store: Any) -> bool:
    db_path = getattr(sql_store, "db_path", None)
    if db_path is None:
        return False
    try:
        return Path(db_path).resolve(strict=False) == DB_PATH.resolve(strict=False)
    except Exception:
        return False


def _should_include_local_review_staging(
    sql_store: Any,
    include_local_review_staging: bool | None,
) -> bool:
    if include_local_review_staging is not None:
        return bool(include_local_review_staging)
    return _uses_default_mkb_path(sql_store)


def _read_r23_review_staging(
    *,
    include_local_review_staging: bool,
    tier_filter: str,
    fact_type_filter: str,
    limit: int,
) -> dict[str, Any]:
    empty = {
        "available": False,
        "count": 0,
        "active_verified_count": 0,
        "all_review_required": True,
        "package_type_counts": {},
        "rows": [],
        "source_classification": "none",
    }
    if not include_local_review_staging:
        return empty
    db_path = default_r23_review_staging_db_path()
    if db_path is None or not db_path.is_file():
        return empty

    try:
        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            table_exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (R23_STAGING_TABLE,),
            ).fetchone()
            if not table_exists:
                return {**empty, "source_classification": "sqlite_db_missing_table"}

            r23_where = "WHERE imported_by_block=?"
            r23_params: list[Any] = [R23_IMPORTED_BLOCK]
            count = int(
                conn.execute(
                    f"SELECT COUNT(*) FROM {R23_STAGING_TABLE} {r23_where}",
                    r23_params,
                ).fetchone()[0]
            )
            active_verified = int(
                conn.execute(
                    f"""
                    SELECT COUNT(*) FROM {R23_STAGING_TABLE}
                    {r23_where} AND verified=1 AND tier='active'
                    """,
                    r23_params,
                ).fetchone()[0]
            )
            unsafe = int(
                conn.execute(
                    f"""
                    SELECT COUNT(*) FROM {R23_STAGING_TABLE}
                    {r23_where}
                    AND (review_required<>1 OR verified<>0 OR auto_accepted<>0)
                    """,
                    r23_params,
                ).fetchone()[0]
            )
            package_type_counts = {
                str(row[0]): int(row[1])
                for row in conn.execute(
                    f"""
                    SELECT package_type, COUNT(*) FROM {R23_STAGING_TABLE}
                    {r23_where}
                    GROUP BY package_type
                    ORDER BY package_type
                    """,
                    r23_params,
                ).fetchall()
            }

            row_clauses = ["imported_by_block=?"]
            row_params: list[Any] = [R23_IMPORTED_BLOCK]
            if tier_filter == "review_bound":
                row_clauses.append("review_required=1")
            elif tier_filter != "all":
                row_clauses.append("tier=?")
                row_params.append(tier_filter)
            if fact_type_filter != "all":
                row_clauses.append("package_type=?")
                row_params.append(fact_type_filter)
            row_where = " WHERE " + " AND ".join(row_clauses)
            rows = conn.execute(
                f"""
                SELECT staging_id, package_type, tier, status, review_required, created_at
                FROM {R23_STAGING_TABLE}
                {row_where}
                ORDER BY created_at DESC, staging_id DESC
                LIMIT ?
                """,
                [*row_params, int(limit)],
            ).fetchall()
    except sqlite3.Error:
        return {**empty, "source_classification": "sqlite_db_unreadable"}

    public_rows = [
        {
            "record_id": safe_record_id(row["staging_id"]),
            "record_id_full": str(row["staging_id"]),
            "fact_type": str(row["package_type"]),
            "specialty": "general",
            "specialty_label": specialty_label("general"),
            "tier": str(row["tier"]),
            "status": str(row["status"]),
            "requires_review": bool(row["review_required"]),
            "operator_review_status": "review_required",
            "display_content": f"R23 {str(row['package_type']).replace('_', ' ')} record awaiting review",
            "source_scope": "local_review_staging",
            "package_type": str(row["package_type"]),
        }
        for row in rows
    ]
    return {
        "available": True,
        "count": count,
        "active_verified_count": active_verified,
        "all_review_required": unsafe == 0,
        "package_type_counts": package_type_counts,
        "rows": public_rows,
        "source_classification": "sqlite_db",
    }


def build_mkb_explorer_model(
    sql_store: Any,
    *,
    specialty_filter: str = "all",
    tier_filter: str = "all",
    fact_type_filter: str = "all",
    limit: int = 50,
    include_local_review_staging: bool | None = None,
) -> dict[str, Any]:
    """Return counts, filter metadata, and public-safe record rows."""
    specialty = str(specialty_filter or "all")
    if specialty != "all":
        specialty = validate_specialty_key(specialty)
    tier = str(tier_filter or "all")
    fact_type = str(fact_type_filter or "all")
    include_staging = _should_include_local_review_staging(sql_store, include_local_review_staging)
    staging_model = _read_r23_review_staging(
        include_local_review_staging=include_staging,
        tier_filter=tier,
        fact_type_filter=fact_type,
        limit=limit,
    )

    if sql_store is None:
        return {
            "available": bool(staging_model["available"]),
            "counts": {
                "total": staging_model["count"],
                "active": 0,
                "quarantined": 0,
                "review_bound": staging_model["count"],
                "superseded": 0,
                "active_verified": staging_model["active_verified_count"],
                "review_required_staging": staging_model["count"],
                "r23_imported": staging_model["count"],
                "r23_package_type_counts": staging_model["package_type_counts"],
            },
            "rows": staging_model["rows"],
            "row_count": len(staging_model["rows"]),
            "filters": {"specialty": specialty, "tier": tier, "fact_type": fact_type},
            "tier_status_visible": True,
            "local_review_staging": staging_model,
        }

    counts = {
        "total": _row_count(sql_store),
        "active": _row_count(sql_store, " WHERE tier=?", ["active"]),
        "quarantined": _row_count(sql_store, " WHERE tier=?", ["quarantined"]),
        "review_bound": _row_count(sql_store, " WHERE requires_review=1"),
        "superseded": _row_count(sql_store, " WHERE tier=?", ["superseded"]),
    }
    counts["active_verified"] = counts["active"]
    counts["review_required_staging"] = staging_model["count"]
    counts["r23_imported"] = staging_model["count"]
    counts["r23_package_type_counts"] = staging_model["package_type_counts"]
    if staging_model["available"]:
        counts["total"] += staging_model["count"]
        counts["review_bound"] += staging_model["count"]

    clauses: list[str] = []
    params: list[Any] = []
    if specialty != "all":
        clauses.append("specialty=?")
        params.append(specialty)
    if tier != "all":
        if tier == "review_bound":
            clauses.append("requires_review=1")
        else:
            clauses.append("tier=?")
            params.append(tier)
    if fact_type != "all":
        clauses.append("fact_type=?")
        params.append(fact_type)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    query = f"SELECT * FROM records{where} ORDER BY first_recorded DESC LIMIT ?"
    with sql_store._get_conn() as conn:
        rows = conn.execute(query, [*params, int(limit)]).fetchall()
    records = [sql_store._row_to_record(row) for row in rows]

    public_rows: list[dict[str, Any]] = []
    for record in records:
        structured = record.structured or {}
        public_rows.append(
            {
                "record_id": safe_record_id(record.id),
                "record_id_full": record.id,
                "fact_type": record.fact_type,
                "specialty": validate_specialty_key(record.specialty),
                "specialty_label": specialty_label(record.specialty),
                "tier": record.tier,
                "status": record.status,
                "requires_review": bool(record.requires_review),
                "operator_review_status": str(structured.get("operator_review_status") or ""),
                "display_content": display_content_public_safe(record),
                "source_scope": "active_mkb",
            }
        )
    public_rows.extend(staging_model["rows"])

    return {
        "available": True,
        "counts": counts,
        "rows": public_rows,
        "row_count": len(public_rows),
        "filters": {
            "specialty": specialty,
            "tier": tier,
            "fact_type": fact_type,
        },
        "tier_status_visible": all(
            {"tier", "status", "requires_review"}.issubset(row.keys()) for row in public_rows
        )
        if public_rows
        else True,
        "local_review_staging": staging_model,
    }


__all__ = [
    "build_mkb_explorer_model",
    "default_r23_review_staging_db_path",
    "display_content_public_safe",
    "safe_record_id",
]
