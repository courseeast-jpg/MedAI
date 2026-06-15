"""Private/local-only MKB review staging payload reader.

The public MKB Explorer can show counts and review metadata. This module adds a
detail model for local review of R23/R25 staging rows without promoting records
to active MKB and without committing raw source text.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from app.mkb_explorer_model import R23_IMPORTED_BLOCK, R23_STAGING_TABLE, default_r23_review_staging_db_path


PAYLOAD_TABLE = "mkb_review_staging_payloads"
SOURCE_EVIDENCE_TABLE = "mkb_review_staging_source_evidence"
QUALITY_TABLE = "mkb_review_staging_quality_metrics"
COVERAGE_TABLE = "mkb_review_staging_corpus_coverage"


def _private_root() -> Path | None:
    local_app_data = os.getenv("LOCALAPPDATA")
    if not local_app_data:
        return None
    return Path(local_app_data) / "MedAI_Private"


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or default_r23_review_staging_db_path()
    if path is None:
        raise FileNotFoundError("R23 staging DB path is not available")
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def _json_loads(value: str | None, fallback: Any) -> Any:
    try:
        return json.loads(value or "")
    except Exception:
        return fallback


def get_staging_detail(record_id: str, *, db_path: Path | None = None) -> dict[str, Any]:
    """Return public-safe private detail metadata for a staging record."""
    with _connect(db_path) as conn:
        record = conn.execute(
            f"SELECT * FROM {R23_STAGING_TABLE} WHERE staging_id=?",
            (record_id,),
        ).fetchone()
        if record is None:
            return {"available": False, "record_id": record_id}
        payload = conn.execute(
            f"SELECT * FROM {PAYLOAD_TABLE} WHERE record_id=?",
            (record_id,),
        ).fetchone()
        evidence = conn.execute(
            f"SELECT * FROM {SOURCE_EVIDENCE_TABLE} WHERE record_id=?",
            (record_id,),
        ).fetchone()
        quality = conn.execute(
            f"SELECT * FROM {QUALITY_TABLE} WHERE record_id=?",
            (record_id,),
        ).fetchone()
        coverage = conn.execute(
            f"SELECT * FROM {COVERAGE_TABLE} WHERE record_id=?",
            (record_id,),
        ).fetchone()

    payload_available = bool(payload and payload["payload_available"])
    return {
        "available": True,
        "record_id": record["staging_id"],
        "safe_doc_id": record["document_id"],
        "corpus_id": coverage["corpus_id"] if coverage else "unknown",
        "package_type": record["package_type"],
        "source_phase": record["source_phase"],
        "review_status": {
            "review_required": bool(record["review_required"]),
            "verified": bool(record["verified"]),
            "auto_accepted": bool(record["auto_accepted"]),
            "status": record["status"],
            "tier": record["tier"],
        },
        "payload_available": payload_available,
        "structured_payload": _json_loads(payload["structured_payload_json"], {}) if payload else {},
        "extracted_items": _json_loads(payload["extracted_items_json"], []) if payload else [],
        "extracted_sections": _json_loads(payload["extracted_sections_json"], []) if payload else [],
        "quality_metrics": {
            "section_count": int(quality["section_count"]) if quality else int(record["section_count"]),
            "item_count": int(quality["item_count"]) if quality else 0,
            "warning_count": int(quality["warning_count"]) if quality else int(record["warnings_count"]),
            "schema_valid": bool(quality["schema_valid"]) if quality else False,
            "minimal_review": bool(quality["minimal_review"]) if quality else False,
        },
        "source_evidence": {
            "preview_available": bool(evidence["preview_available"]) if evidence else False,
            "evidence_type": evidence["evidence_type"] if evidence else "unavailable",
            "preview_char_count": int(evidence["preview_char_count"]) if evidence else 0,
            "page_count": int(evidence["page_count"]) if evidence and evidence["page_count"] is not None else None,
            "public_preview_allowed": bool(evidence["public_preview_allowed"]) if evidence else False,
        },
        "terminal_reason": record["reason_code"],
        "document_state": coverage["document_state"] if coverage else "review_only_reason_available",
        "active_verified_promotion_allowed": False,
        "auto_accept_allowed": False,
        "medical_decision_allowed": False,
    }


def get_private_source_preview(
    record_id: str,
    *,
    max_chars: int = 1200,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Read local-private source preview text for an operator.

    This function is intentionally opt-in and returns raw text only to the local
    process. Public reports and tests must not serialize its return payload.
    """
    root = _private_root()
    if root is None:
        return {"available": False, "preview": ""}
    with _connect(db_path) as conn:
        evidence = conn.execute(
            f"SELECT * FROM {SOURCE_EVIDENCE_TABLE} WHERE record_id=? AND preview_available=1",
            (record_id,),
        ).fetchone()
    if evidence is None or not evidence["private_artifact_ref"]:
        return {"available": False, "preview": ""}
    ref = str(evidence["private_artifact_ref"])
    candidate = (root / ref).resolve(strict=False)
    try:
        candidate.relative_to(root.resolve(strict=False))
    except ValueError:
        return {"available": False, "preview": ""}
    if not candidate.is_file():
        return {"available": False, "preview": ""}
    text = candidate.read_text(encoding="utf-8", errors="ignore")[: int(max_chars)]
    return {
        "available": True,
        "preview": text,
        "preview_char_count": len(text),
        "public_preview_allowed": False,
    }


def build_staging_quality_view(record_id: str, *, db_path: Path | None = None) -> dict[str, Any]:
    detail = get_staging_detail(record_id, db_path=db_path)
    if not detail.get("available"):
        return detail
    return {
        "available": True,
        "record_id": detail["record_id"],
        "safe_doc_id": detail["safe_doc_id"],
        "package_type": detail["package_type"],
        "document_state": detail["document_state"],
        "quality_metrics": detail["quality_metrics"],
        "payload_available": detail["payload_available"],
        "source_preview_available": detail["source_evidence"]["preview_available"],
        "terminal_reason": detail["terminal_reason"],
    }


__all__ = [
    "COVERAGE_TABLE",
    "PAYLOAD_TABLE",
    "QUALITY_TABLE",
    "SOURCE_EVIDENCE_TABLE",
    "build_staging_quality_view",
    "get_private_source_preview",
    "get_staging_detail",
]
