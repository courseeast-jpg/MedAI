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


_TOKEN_RE = __import__("re").compile(r"\[[A-Z_]+_\d+\]")


def _item_to_text(item: Any) -> str:
    """Render one extracted item/fact as a compact readable line (no JSON braces)."""
    if isinstance(item, dict):
        parts = []
        for key, value in item.items():
            if value in (None, "", [], {}):
                continue
            if isinstance(value, (list, dict)):
                value = json.dumps(value, ensure_ascii=False)
            parts.append(f"{key}: {value}")
        return " · ".join(parts) if parts else ""
    if isinstance(item, (list, tuple)):
        return " · ".join(_item_to_text(part) for part in item if part not in (None, ""))
    return str(item).strip()


def _section_name(section: Any, fallback: str) -> str:
    if isinstance(section, dict):
        return str(section.get("section") or section.get("name") or fallback)
    return fallback


def _section_items(section: Any) -> list[Any]:
    if isinstance(section, dict):
        items = section.get("items")
        if isinstance(items, list):
            return items
    return []


def build_readable_markdown(structured_payload: Any, sections: Any, items: Any) -> tuple[str, int, int, int]:
    """Build operator-readable markdown from a staging payload. Returns
    (markdown, sections_rendered, items_rendered, nonplaceholder_char_count)."""
    lines: list[str] = []
    sections_rendered = 0
    items_rendered = 0
    section_list = sections if isinstance(sections, list) else []
    for idx, section in enumerate(section_list):
        name = _section_name(section, f"section_{idx + 1}")
        sec_items = _section_items(section)
        rendered = [text for text in (_item_to_text(it) for it in sec_items) if text]
        lines.append(f"**{name}**")
        if rendered:
            lines.extend(f"- {text}" for text in rendered)
            items_rendered += len(rendered)
        else:
            lines.append("- (no items in this section; review-bound)")
        sections_rendered += 1
        lines.append("")
    flat_items = items if isinstance(items, list) else []
    flat_rendered = [text for text in (_item_to_text(it) for it in flat_items) if text]
    if flat_rendered:
        lines.append("**Extracted facts**")
        lines.extend(f"- {text}" for text in flat_rendered)
        items_rendered += len(flat_rendered)
        lines.append("")
    if not section_list and not flat_rendered and isinstance(structured_payload, dict) and structured_payload:
        # Minimal/review-bound packages: render the structured payload keys readably.
        for key, value in structured_payload.items():
            if isinstance(value, (list, dict)):
                value = json.dumps(value, ensure_ascii=False)
            lines.append(f"- {key}: {value}")
    markdown = "\n".join(lines).strip()
    stripped = _TOKEN_RE.sub("", markdown)
    nonplaceholder = sum(1 for ch in stripped if not ch.isspace() and ch not in "-*#`>")
    return markdown, sections_rendered, items_rendered, nonplaceholder


def _not_extracted_explanation(detail: dict[str, Any], row: dict[str, Any]) -> str:
    reason = str(detail.get("terminal_reason") or "unknown")
    package_type = str(detail.get("package_type") or "")
    if package_type == "non_sendable_excluded":
        return ("Source is a non-sendable container (e.g. RTF/signal container). It was never "
                f"sent for extraction. Terminal reason: {reason}.")
    if row.get("source_unavailable"):
        return (f"No extracted payload. Terminal reason: {reason}. "
                f"{row.get('source_unavailable_reason') or 'source preview unavailable'}.")
    return (f"No extracted payload. Terminal reason: {reason}. Source evidence resolution: "
            f"{row.get('source_resolution') or 'unknown'}.")


def readable_record_view(record_id: str, *, include_private_preview: bool = False) -> dict[str, Any]:
    """Operator-readable detail for one staging record: readable extracted content,
    sections, items/facts, source-evidence availability, and QA status. Raw clinical text
    is returned ONLY for local UI use; callers must not serialize it to public artifacts.
    `proof_metrics` carries content-free counts/booleans safe for proof evidence."""
    detail = get_comparator_record_detail(record_id, include_private_preview=include_private_preview)
    if not detail.get("available"):
        return {"available": False, "record_id": record_id}
    row = detail.get("qa_row") or {}
    sections = detail.get("extracted_sections") or []
    items = detail.get("extracted_items") or []
    payload = detail.get("structured_payload") or {}
    is_extracted = bool(detail.get("payload_available"))
    markdown, sec_n, item_n, nonplaceholder = build_readable_markdown(payload, sections, items)
    source = dict(detail.get("source_evidence") or {})
    quality = dict(detail.get("quality_metrics") or {})
    warnings = []
    if isinstance(payload, dict):
        raw_warn = payload.get("warnings") or payload.get("extraction_warnings")
        if isinstance(raw_warn, list):
            warnings = [str(w) for w in raw_warn]
    source_visible = bool(source) or bool(row.get("source_resolution"))
    return {
        "available": True,
        "record_id": detail["record_id"],
        "safe_doc_id": detail["safe_doc_id"],
        "corpus_id": detail["corpus_id"],
        "package_type": detail["package_type"],
        "is_extracted": is_extracted,
        "headings": [
            "Extracted content",
            "Extracted sections",
            "Extracted items / facts",
            "Source evidence / original preview",
            "QA decision",
        ],
        "extracted_content_markdown": markdown,
        "sections_readable": [
            {"section": _section_name(s, f"section_{i + 1}"),
             "item_count": len(_section_items(s))}
            for i, s in enumerate(sections)
        ],
        "items_readable": [text for text in (_item_to_text(it) for it in items) if text],
        "warnings": warnings,
        "quality_metrics": quality,
        "source_evidence": {
            "preview_available": bool(source.get("preview_available")),
            "evidence_type": source.get("evidence_type", "unavailable"),
            "page_count": source.get("page_count"),
            "source_resolution": row.get("source_resolution"),
            "source_unavailable": bool(row.get("source_unavailable")),
        },
        "terminal_reason": detail.get("terminal_reason"),
        "failure_bucket": row.get("failure_bucket") or detail.get("terminal_reason"),
        "not_extracted_explanation": "" if is_extracted else _not_extracted_explanation(detail, row),
        "qa_status": row.get("qa_status", "not_reviewed"),
        "proof_metrics": {
            "content_heading_present": True,
            "is_extracted": is_extracted,
            "sections_rendered": sec_n,
            "items_rendered": item_n,
            "nonplaceholder_chars": nonplaceholder,
            "readable_present": bool(markdown),
            "source_evidence_visible": source_visible,
            "terminal_reason_present": bool(detail.get("terminal_reason")),
        },
        "active_verified_promotion_allowed": False,
        "auto_accept_allowed": False,
        "medical_decision_allowed": False,
    }


def representative_proof_records(*, db_path: Path | None = None) -> dict[str, str | None]:
    """Pick one representative record per type for the live-UI readable-render proof."""
    model = build_all_records_qa_comparator(db_path=db_path)
    full_schema = next((r["record_id"] for r in model["extracted_queue"]
                        if r["package_type"] == "full_schema"), None)
    minimal = next((r["record_id"] for r in model["extracted_queue"]
                    if r["package_type"] == "minimal_review_bound"), None)
    not_extracted = next((r["record_id"] for r in model["not_extracted_queue"]), None)
    return {"full_schema": full_schema, "minimal_review": minimal, "not_extracted": not_extracted}


__all__ = [
    "QA_DECISION_TABLE",
    "QA_STATUSES",
    "build_all_records_qa_comparator",
    "build_readable_markdown",
    "create_private_all_records_export",
    "get_comparator_record_detail",
    "readable_record_view",
    "representative_proof_records",
    "save_qa_status",
]
