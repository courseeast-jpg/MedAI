"""Tests for R26 all-staging-records payload/source QA comparator."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from app.mkb_all_records_qa_comparator import (
    QA_DECISION_TABLE,
    apply_qa_filters,
    build_all_records_qa_comparator,
    get_comparator_record_detail,
    save_qa_status,
)
from app.mkb_explorer_model import default_r23_review_staging_db_path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_r26_all_staging_records_payload_source_qa_comparator"


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def _conn() -> sqlite3.Connection:
    db_path = default_r23_review_staging_db_path()
    assert db_path is not None and db_path.is_file()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def test_all_record_qa_queues_cover_complete_staging_set() -> None:
    model = build_all_records_qa_comparator()
    assert model["counts"]["total_staging_records"] == 496
    assert model["counts"]["extracted_payload_records"] == 179
    assert model["counts"]["not_extracted_records"] == 317
    assert len(model["all_extracted_record_ids"]) == 179
    assert len(model["all_not_extracted_record_ids"]) == 317
    assert len(model["extracted_queue"]) == 179
    assert len(model["not_extracted_queue"]) == 317


def test_required_filters_are_supported() -> None:
    model = build_all_records_qa_comparator()
    options = set(model["filter_options"])
    for option in [
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
    ]:
        assert option in options
    assert len(apply_qa_filters(model["rows"], ["Full schema"])) == 161
    assert len(apply_qa_filters(model["rows"], ["Minimal review"])) == 18
    assert len(apply_qa_filters(model["rows"], ["Non-sendable excluded"])) == 2


def test_sample_payloads_and_failure_reasons_are_visible() -> None:
    model = build_all_records_qa_comparator()
    full_id = next(row["record_id"] for row in model["rows"] if row["package_type"] == "full_schema")
    minimal_id = next(row["record_id"] for row in model["rows"] if row["package_type"] == "minimal_review_bound")
    failed_id = next(row["record_id"] for row in model["rows"] if row["package_type"] == "review_only_finalized")
    excluded_id = next(row["record_id"] for row in model["rows"] if row["package_type"] == "non_sendable_excluded")
    full = get_comparator_record_detail(full_id)
    minimal = get_comparator_record_detail(minimal_id)
    failed = get_comparator_record_detail(failed_id)
    excluded = get_comparator_record_detail(excluded_id)
    assert full["payload_available"] is True
    assert full["extracted_sections"]
    assert full["extracted_items"]
    assert full["quality_metrics"]["schema_valid"] is True
    assert minimal["payload_available"] is True
    assert minimal["quality_metrics"]["minimal_review"] is True
    assert failed["payload_available"] is False
    assert failed["terminal_reason"]
    assert excluded["terminal_reason"] == "excluded_rtf_signal_container"


def test_source_resolution_attempted_for_all_and_qa_status_saves() -> None:
    model = build_all_records_qa_comparator()
    assert model["counts"]["source_evidence_refs"] == 496
    assert model["counts"]["source_preview_available"] == 331
    assert model["counts"]["source_unavailable"] == 165
    extracted_id = model["all_extracted_record_ids"][0]
    not_extracted_id = model["all_not_extracted_record_ids"][0]
    assert save_qa_status(extracted_id, "looks_correct")["saved"] is True
    assert save_qa_status(not_extracted_id, "not_extracted_reviewed")["saved"] is True
    with _conn() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
            (QA_DECISION_TABLE,),
        ).fetchone()[0] == 1
        unsafe = conn.execute(
            f"SELECT COUNT(*) FROM {QA_DECISION_TABLE} WHERE active_mkb_write<>0 OR verified_promotion<>0 OR auto_accept<>0"
        ).fetchone()[0]
    assert unsafe == 0


def test_summary_required_fields_are_passed() -> None:
    s = _summary()
    assert s["overall_result"] == "PASS"
    assert s["provider_model_call_made"] is False
    assert s["live_extraction_started"] is False
    assert s["all_staging_records_indexed"] == 496
    assert s["extracted_payload_records_indexed"] == 179
    assert s["not_extracted_records_indexed"] == 317
    assert s["source_evidence_resolution_attempted"] == 496
    assert s["source_preview_available_count"] == 331
    assert s["all_extracted_records_selectable"] is True
    assert s["all_not_extracted_records_selectable"] is True
    assert s["qa_comparator_created"] is True
    assert s["extracted_payload_queue_created"] is True
    assert s["not_extracted_failure_queue_created"] is True
    assert s["qa_decision_store_created"] is True
    assert s["qa_status_save_for_extracted_verified"] is True
    assert s["qa_status_save_for_not_extracted_verified"] is True
    assert s["active_verified_records_created"] == 0
    assert s["auto_accept_enabled"] is False
    assert s["medical_decision_made"] is False
    assert s["privacy_result"] == "passed"
    assert s["safety_result"] == "passed"


def test_public_reports_have_no_private_paths_or_payloads() -> None:
    forbidden = ["C:\\", "GEMINI_API_KEY", "Bearer ", "Authorization:", "ya29.", "raw_provider_response"]
    for path in REPORT_DIR.glob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for marker in forbidden:
            assert marker not in text
