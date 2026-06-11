"""Local adapter fallback for degraded Run & Review.

MEDAI-UI-ADAPTER-FALLBACK-RUN-REVIEW-10.

This module is intentionally independent of ExecutionPipeline. It is used only
when startup left SQLite available but ``execution`` is None. It accepts raw
text or TXT files, extracts deterministic lab observations, persists them as
review-bound MKB records, and returns a public-safe payload for the existing
Run & Review preview card.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config import TIER_QUARANTINED, TRUST_CLINICAL
from app.extracted_information_preview import build_extracted_information_preview_plan
from app.operator_review_actions import (
    accept_after_source_comparison,
    defer_extracted_fact,
    reject_extracted_fact,
)
from app.schemas import MKBRecord
from app.specialty_selection import specialty_label, validate_specialty_key
from execution.extracted_medical_facts import (
    CONSERVATIVE_CONFIDENCE,
    extract_lab_observation_entities,
    summarize_extracted_facts_for_public_report,
)

ADAPTER_FALLBACK_MODE = "ui_adapter_fallback_run_review"
ADAPTER_FALLBACK_SOURCE_NAME = "adapter-fallback-run-review"
SUPPORTED_TEXT_SUFFIXES = {".txt"}


def safe_text_handle(*, size: int, suffix: str = ".txt") -> str:
    """Return a public-safe input handle derived only from size and suffix."""
    clean_suffix = (suffix or ".txt").lower()
    digest = hashlib.sha256(f"{int(size)}:{clean_suffix}".encode("utf-8")).hexdigest()
    return f"adapter_fallback_{digest[:10]}"


def _read_text_input(*, raw_text: str | None = None, txt_path: Path | str | None = None) -> tuple[str, str]:
    if raw_text is not None:
        return str(raw_text), safe_text_handle(size=len(str(raw_text).encode("utf-8")), suffix=".txt")
    if txt_path is None:
        raise ValueError("raw_text or txt_path is required")
    path = Path(txt_path)
    if path.suffix.lower() not in SUPPORTED_TEXT_SUFFIXES:
        raise ValueError("adapter fallback only accepts TXT input")
    text = path.read_text(encoding="utf-8", errors="replace")
    return text, safe_text_handle(size=path.stat().st_size, suffix=path.suffix)


def _record_state(record: MKBRecord) -> dict[str, Any]:
    return {
        "fact_type": record.fact_type,
        "tier": record.tier,
        "status": record.status,
        "requires_review": bool(record.requires_review),
        "ddi_status": record.ddi_status or "",
    }


def process_adapter_fallback_run_review(
    sql_store: Any,
    *,
    raw_text: str | None = None,
    txt_path: Path | str | None = None,
    specialty: str = "general",
    selected_specialty: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Run the degraded local adapter path and persist review-bound records."""
    if sql_store is None:
        raise ValueError("SQLite store is required for adapter fallback")
    selected_specialty_key = validate_specialty_key(selected_specialty or specialty)
    text, input_handle = _read_text_input(raw_text=raw_text, txt_path=txt_path)
    session = session_id or f"adapter-fallback-{uuid4()}"
    metadata = {
        "document_type": "lab_report",
        "document_family_classification_diagnostic": {"candidate_family": "Lab result"},
    }
    entities = extract_lab_observation_entities(text, metadata)
    summary = summarize_extracted_facts_for_public_report(entities)

    record_ids: list[str] = []
    record_states: dict[str, dict[str, Any]] = {}
    for entity in entities:
        structured = dict(entity.get("structured") or {})
        test_name = str(entity.get("text") or structured.get("test_name") or "Lab observation")
        value = str(structured.get("value") or "")
        unit = str(structured.get("unit") or "")
        content = f"Test: {test_name}: {value} {unit}".rstrip()
        record = MKBRecord(
            fact_type="test_result",
            content=content,
            structured={
                "text": test_name,
                **structured,
                "fallback_mode": ADAPTER_FALLBACK_MODE,
                "source_input_handle": input_handle,
            },
            specialty=selected_specialty_key,
            source_type="extraction",
            source_name=ADAPTER_FALLBACK_SOURCE_NAME,
            trust_level=TRUST_CLINICAL,
            confidence=float(entity.get("confidence", CONSERVATIVE_CONFIDENCE)),
            status="active",
            tier=TIER_QUARANTINED,
            ddi_checked=False,
            ddi_status=None,
            extraction_method="rules_based / adapter_fallback",
            requires_review=True,
            session_id=session,
        )
        sql_store.write_record(record, session_id=session)
        record_ids.append(record.id)
        record_states[record.id] = _record_state(record)

    run_item = {
        "file_name": input_handle,
        "status": "review" if entities else "empty",
        "outcome": "queued_for_review" if entities else "empty",
        "selected_extractor": ADAPTER_FALLBACK_MODE,
        "confidence": CONSERVATIVE_CONFIDENCE if entities else 0.0,
        "validation_status": "queued_for_review" if entities else "empty",
        "document_type": "Lab result" if entities else "Unknown",
        "ocr_quality_band": "txt_input",
        "language_text_visibility": "readable_text",
        "cyrillic_ocr_recommended": False,
        "ocr_gate_review_only": True,
        "ocr_gate_auto_accept_allowed": False,
        "document_family_classification_diagnostic": {"candidate_family": "Lab result"},
        "operator_review_reason": "adapter_fallback_review_required",
        "operator_reason_label": "Review required",
        "extracted_medical_facts_preview_safe": summary["extracted_medical_facts_preview_safe"],
        "extracted_medical_fact_count": int(summary["extracted_medical_fact_count"]),
        "extracted_medical_fact_types": list(summary["extracted_medical_fact_types"]),
        "extraction_to_mkb_candidate_count": int(summary["extracted_medical_fact_count"]),
        "extraction_to_mkb_written_count": 0,
        "extraction_to_mkb_review_count": len(record_ids),
        "extracted_medical_fact_record_ids": record_ids,
        "extracted_medical_fact_record_states": record_states,
        "external_api_used": False,
        "auto_accept_allowed": False,
        "adapter_fallback_mode": ADAPTER_FALLBACK_MODE,
        "input_safe_handle": input_handle,
        "selected_specialty": selected_specialty_key,
        "selected_specialty_label": specialty_label(selected_specialty_key),
    }
    preview_plan = build_extracted_information_preview_plan(run_item)
    return {
        "mode": ADAPTER_FALLBACK_MODE,
        "ready": bool(sql_store is not None),
        "run_item": run_item,
        "preview_plan": preview_plan,
        "structured_facts_extracted": int(summary["extracted_medical_fact_count"]),
        "review_bound_records_persisted": len(record_ids),
        "ui_preview_rows": int(preview_plan.get("row_count") or 0),
        "selected_specialty": selected_specialty_key,
        "selected_specialty_label": specialty_label(selected_specialty_key),
        "external_api_used": False,
        "auto_accept_enabled": False,
    }


def run_operator_action_proof(sql_store: Any, record_ids: list[str], *, session_id: str = "adapter-fallback-proof") -> int:
    """Run synthetic accept/reject/defer proof against the first three records."""
    if len(record_ids) < 3:
        return 0
    results = [
        accept_after_source_comparison(sql_store, record_ids[0], operator_note="synthetic verify", session_id=session_id),
        reject_extracted_fact(sql_store, record_ids[1], session_id=session_id),
        defer_extracted_fact(sql_store, record_ids[2], session_id=session_id),
    ]
    return sum(1 for result in results if result.success)


__all__ = [
    "ADAPTER_FALLBACK_MODE",
    "ADAPTER_FALLBACK_SOURCE_NAME",
    "SUPPORTED_TEXT_SUFFIXES",
    "process_adapter_fallback_run_review",
    "run_operator_action_proof",
    "safe_text_handle",
]
