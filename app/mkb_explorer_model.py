"""Streamlit-free MKB Explorer model helpers.

MEDAI-UI-CAPABILITY-RESTORE-11B.

The model exposes public-safe record metadata for UI rendering and validation.
It reads only through the provided SQLite store and never emits raw source text,
raw filenames, private paths, or runtime DB rows.
"""
from __future__ import annotations

from typing import Any

from app.schemas import MKBRecord
from app.specialty_selection import specialty_label, validate_specialty_key


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


def build_mkb_explorer_model(
    sql_store: Any,
    *,
    specialty_filter: str = "all",
    tier_filter: str = "all",
    fact_type_filter: str = "all",
    limit: int = 50,
) -> dict[str, Any]:
    """Return counts, filter metadata, and public-safe record rows."""
    if sql_store is None:
        return {
            "available": False,
            "counts": {"total": 0, "active": 0, "quarantined": 0, "review_bound": 0, "superseded": 0},
            "rows": [],
            "row_count": 0,
            "filters": {},
            "tier_status_visible": True,
        }

    specialty = str(specialty_filter or "all")
    if specialty != "all":
        specialty = validate_specialty_key(specialty)
    tier = str(tier_filter or "all")
    fact_type = str(fact_type_filter or "all")

    counts = {
        "total": _row_count(sql_store),
        "active": _row_count(sql_store, " WHERE tier=?", ["active"]),
        "quarantined": _row_count(sql_store, " WHERE tier=?", ["quarantined"]),
        "review_bound": _row_count(sql_store, " WHERE requires_review=1"),
        "superseded": _row_count(sql_store, " WHERE tier=?", ["superseded"]),
    }

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
            }
        )

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
    }


__all__ = [
    "build_mkb_explorer_model",
    "display_content_public_safe",
    "safe_record_id",
]
