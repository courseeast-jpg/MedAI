"""All-record local QA comparator for MKB review staging rows."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from app.mkb_explorer_model import R23_IMPORTED_BLOCK, R23_STAGING_TABLE, safe_record_id
from app.mkb_staging_payload_reader import (
    COVERAGE_TABLE,
    PAYLOAD_TABLE,
    QUALITY_TABLE,
    SOURCE_EVIDENCE_TABLE,
    default_r23_review_staging_db_path,
    get_private_source_preview,
    get_staging_detail,
)


QA_DECISION_TABLE = "mkb_review_staging_qa_decisions"
QA_STATUSES = {
    "not_reviewed",
    "looks_correct",
    "partly_correct",
    "incorrect",
    "source_unavailable",
    "needs_manual_review",
    "not_extracted_reviewed",
}


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or default_r23_review_staging_db_path()
    if path is None:
        raise FileNotFoundError("R23 staging DB path is unavailable")
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def _private_output_dir() -> Path | None:
    local_app_data = os.getenv("LOCALAPPDATA")
    if not local_app_data:
        return None
    return Path(local_app_data) / "MedAI_Private" / "mkb" / "r26_all_records_qa_comparator"


def ensure_qa_decision_store(*, db_path: Path | None = None) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {QA_DECISION_TABLE} (
                record_id TEXT PRIMARY KEY,
                safe_doc_id TEXT NOT NULL,
                corpus_id TEXT NOT NULL,
                qa_status TEXT NOT NULL,
                qa_note TEXT NOT NULL,
                reviewer TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                active_mkb_write INTEGER NOT NULL,
                verified_promotion INTEGER NOT NULL,
                auto_accept INTEGER NOT NULL
            )
            """
        )
        conn.commit()


def source_resolution_for_row(row: sqlite3.Row) -> dict[str, Any]:
    preview_available = bool(row["preview_available"])
    evidence_type = str(row["evidence_type"] or "unavailable")
    ref_available = bool(str(row["private_artifact_ref"] or ""))
    if evidence_type == "pdf_page_ref" and ref_available:
        resolution = "pdf_reference"
    elif evidence_type == "rendered_page_ref" and ref_available:
        resolution = "rendered_page_reference"
    elif preview_available:
        resolution = "ocr_or_source_preview"
    elif ref_available:
        resolution = "source_ref_only"
    else:
        resolution = "source_unavailable"
    reason = "" if resolution != "source_unavailable" else "no_private_source_preview_or_pdf_reference_available"
    return {
        "source_resolution": resolution,
        "source_preview_available": preview_available,
        "source_unavailable": resolution == "source_unavailable",
        "source_unavailable_reason": reason,
        "evidence_type": evidence_type,
        "preview_char_count": int(row["preview_char_count"] or 0),
        "page_count": row["page_count"],
        "public_preview_allowed": bool(row["public_preview_allowed"]),
    }


def _base_rows(*, db_path: Path | None = None) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT
                r.staging_id,
                r.document_id,
                r.package_type,
                r.source_phase,
                r.terminal_state,
                r.reason_code,
                r.review_required,
                r.verified,
                r.auto_accepted,
                c.corpus_id,
                c.document_state,
                c.payload_available,
                c.source_preview_available,
                q.section_count,
                q.item_count,
                q.warning_count,
                q.schema_valid,
                q.minimal_review,
                e.evidence_type,
                e.private_artifact_ref,
                e.preview_available,
                e.preview_char_count,
                e.page_count,
                e.public_preview_allowed,
                d.qa_status
            FROM {R23_STAGING_TABLE} r
            JOIN {COVERAGE_TABLE} c ON c.record_id = r.staging_id
            JOIN {QUALITY_TABLE} q ON q.record_id = r.staging_id
            JOIN {SOURCE_EVIDENCE_TABLE} e ON e.record_id = r.staging_id
            LEFT JOIN {QA_DECISION_TABLE} d ON d.record_id = r.staging_id
            WHERE r.imported_by_block=?
            ORDER BY c.payload_available DESC, r.package_type, r.staging_id
            """,
            (R23_IMPORTED_BLOCK,),
        ).fetchall()
    result: list[dict[str, Any]] = []
    for row in rows:
        source = source_resolution_for_row(row)
        result.append(
            {
                "record_id": str(row["staging_id"]),
                "record_id_short": safe_record_id(str(row["staging_id"])),
                "safe_doc_id": str(row["document_id"]),
                "corpus_id": str(row["corpus_id"]),
                "package_type": str(row["package_type"]),
                "source_phase": str(row["source_phase"]),
                "document_state": str(row["document_state"]),
                "terminal_reason": str(row["reason_code"]),
                "failure_bucket": str(row["reason_code"]),
                "payload_available": bool(row["payload_available"]),
                "review_required": bool(row["review_required"]),
                "verified": bool(row["verified"]),
                "auto_accepted": bool(row["auto_accepted"]),
                "quality_metrics": {
                    "section_count": int(row["section_count"] or 0),
                    "item_count": int(row["item_count"] or 0),
                    "warning_count": int(row["warning_count"] or 0),
                    "schema_valid": bool(row["schema_valid"]),
                    "minimal_review": bool(row["minimal_review"]),
                    "payload_available": bool(row["payload_available"]),
                },
                "qa_status": str(row["qa_status"] or "not_reviewed"),
                "future_review_route": _future_review_route(str(row["package_type"]), source),
                **source,
            }
        )
    return result


def _future_review_route(package_type: str, source: dict[str, Any]) -> str:
    if package_type == "non_sendable_excluded":
        return "unsupported_container_handling"
    if source["source_preview_available"]:
        return "manual_review"
    if source["source_resolution"] in {"pdf_reference", "rendered_page_reference"}:
        return "original_pdf_route"
    return "manual_review_source_unavailable"


def apply_qa_filters(rows: Iterable[dict[str, Any]], filters: Iterable[str]) -> list[dict[str, Any]]:
    active = {str(item) for item in filters if str(item) and str(item) != "All"}
    result = list(rows)
    for flt in active:
        if flt == "Extracted only":
            result = [row for row in result if row["payload_available"]]
        elif flt == "Not extracted only":
            result = [row for row in result if not row["payload_available"]]
        elif flt == "Source preview available":
            result = [row for row in result if row["source_preview_available"]]
        elif flt == "Source unavailable":
            result = [row for row in result if row["source_unavailable"]]
        elif flt == "Full schema":
            result = [row for row in result if row["package_type"] == "full_schema"]
        elif flt == "Minimal review":
            result = [row for row in result if row["package_type"] == "minimal_review_bound"]
        elif flt == "Review-only reason":
            result = [row for row in result if row["document_state"] in {"review_only_reason_available", "source_evidence_preview_available"}]
        elif flt == "Non-sendable excluded":
            result = [row for row in result if row["package_type"] == "non_sendable_excluded"]
        elif flt == "Corpus 1":
            result = [row for row in result if row["corpus_id"] == "corpus1"]
        elif flt == "Corpus 2":
            result = [row for row in result if row["corpus_id"] == "corpus2"]
        elif flt == "Warning present":
            result = [row for row in result if row["quality_metrics"]["warning_count"] > 0]
        elif flt == "Low/empty item count":
            result = [row for row in result if row["quality_metrics"]["item_count"] <= 0]
    return result


def build_all_records_qa_comparator(
    *,
    filters: Iterable[str] = ("All",),
    db_path: Path | None = None,
) -> dict[str, Any]:
    ensure_qa_decision_store(db_path=db_path)
    rows = _base_rows(db_path=db_path)
    extracted = [row for row in rows if row["payload_available"]]
    not_extracted = [row for row in rows if not row["payload_available"]]
    filtered = apply_qa_filters(rows, filters)
    source_preview_count = sum(1 for row in rows if row["source_preview_available"])
    source_unavailable_count = sum(1 for row in rows if row["source_unavailable"])
    return {
        "available": True,
        "counts": {
            "total_staging_records": len(rows),
            "extracted_payload_records": len(extracted),
            "not_extracted_records": len(not_extracted),
            "source_evidence_refs": len(rows),
            "source_preview_available": source_preview_count,
            "source_unavailable": source_unavailable_count,
            "corpus1": sum(1 for row in rows if row["corpus_id"] == "corpus1"),
            "corpus2": sum(1 for row in rows if row["corpus_id"] == "corpus2"),
        },
        "rows": filtered,
        "extracted_queue": apply_qa_filters(extracted, filters),
        "not_extracted_queue": apply_qa_filters(not_extracted, filters),
        "all_extracted_record_ids": [row["record_id"] for row in extracted],
        "all_not_extracted_record_ids": [row["record_id"] for row in not_extracted],
        "filter_options": [
            "All",
            "Extracted only",
            "Not extracted only",
            "Source preview available",
            "Source unavailable",
            "Full schema",
            "Minimal review",
            "Review-only reason",
            "Non-sendable excluded",
            "Corpus 1",
            "Corpus 2",
            "Warning present",
            "Low/empty item count",
        ],
        "active_verified_promotion_allowed": False,
        "auto_accept_allowed": False,
        "medical_decision_allowed": False,
    }


def get_comparator_record_detail(record_id: str, *, include_private_preview: bool = False) -> dict[str, Any]:
    detail = get_staging_detail(record_id)
    if not detail.get("available"):
        return detail
    rows = [row for row in _base_rows() if row["record_id"] == record_id]
    row = rows[0] if rows else {}
    source_preview = {"available": False}
    if include_private_preview:
        source_preview = get_private_source_preview(record_id)
    return {
        **detail,
        "qa_row": row,
        "source_preview": source_preview,
        "comparison_controls": {
            "open_detail": True,
            "open_source_evidence": bool(row.get("source_preview_available")),
            "mark_qa_status_locally": True,
        },
        "active_verified_promotion_allowed": False,
        "auto_accept_allowed": False,
        "medical_decision_allowed": False,
    }


def save_qa_status(
    record_id: str,
    qa_status: str,
    *,
    qa_note: str = "",
    reviewer: str = "local_operator",
    db_path: Path | None = None,
) -> dict[str, Any]:
    if qa_status not in QA_STATUSES:
        raise ValueError(f"Unsupported QA status: {qa_status}")
    detail = get_staging_detail(record_id, db_path=db_path)
    if not detail.get("available"):
        raise KeyError(record_id)
    ensure_qa_decision_store(db_path=db_path)
    timestamp = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    with _connect(db_path) as conn:
        conn.execute(
            f"""
            INSERT OR REPLACE INTO {QA_DECISION_TABLE}
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                detail["safe_doc_id"],
                detail["corpus_id"],
                qa_status,
                qa_note,
                reviewer,
                timestamp,
                0,
                0,
                0,
            ),
        )
        conn.commit()
        saved = conn.execute(
            f"SELECT * FROM {QA_DECISION_TABLE} WHERE record_id=?",
            (record_id,),
        ).fetchone()
    return {
        "saved": saved is not None and saved["qa_status"] == qa_status,
        "record_id": record_id,
        "qa_status": qa_status,
        "active_mkb_write": False,
        "verified_promotion": False,
        "auto_accept": False,
    }


def create_private_all_records_export(*, db_path: Path | None = None) -> dict[str, Any]:
    model = build_all_records_qa_comparator(db_path=db_path)
    output_dir = _private_output_dir()
    if output_dir is None:
        return {"created": False, "count": 0}
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "record_id": row["record_id"],
            "safe_doc_id": row["safe_doc_id"],
            "corpus_id": row["corpus_id"],
            "package_type": row["package_type"],
            "document_state": row["document_state"],
            "payload_available": row["payload_available"],
            "source_preview_available": row["source_preview_available"],
            "source_resolution": row["source_resolution"],
            "terminal_reason": row["terminal_reason"],
            "qa_status": row["qa_status"],
        }
        for row in model["rows"]
    ]
    (output_dir / "all_qa_comparator_index_private.json").write_text(
        json.dumps({"records": records}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "all_extracted_payload_index_private.json").write_text(
        json.dumps({"records": [row for row in records if row["payload_available"]]}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "all_not_extracted_failure_index_private.json").write_text(
        json.dumps({"records": [row for row in records if not row["payload_available"]]}, indent=2),
        encoding="utf-8",
    )
    return {"created": True, "count": len(records)}


__all__ = [
    "QA_DECISION_TABLE",
    "QA_STATUSES",
    "build_all_records_qa_comparator",
    "create_private_all_records_export",
    "get_comparator_record_detail",
    "save_qa_status",
]
