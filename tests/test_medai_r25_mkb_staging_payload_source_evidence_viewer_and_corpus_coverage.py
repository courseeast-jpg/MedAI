"""Tests for R25 staging payload/evidence viewer and corpus coverage."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from app.mkb_explorer_model import R23_IMPORTED_BLOCK, default_r23_review_staging_db_path
from app.mkb_staging_payload_reader import (
    COVERAGE_TABLE,
    PAYLOAD_TABLE,
    QUALITY_TABLE,
    SOURCE_EVIDENCE_TABLE,
    build_staging_quality_view,
    get_staging_detail,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_r25_mkb_staging_payload_source_evidence_viewer_and_corpus_coverage"


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def _conn() -> sqlite3.Connection:
    db_path = default_r23_review_staging_db_path()
    assert db_path is not None and db_path.is_file()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def test_private_staging_tables_exist_and_are_idempotent() -> None:
    with _conn() as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert PAYLOAD_TABLE in tables
        assert SOURCE_EVIDENCE_TABLE in tables
        assert QUALITY_TABLE in tables
        assert COVERAGE_TABLE in tables
        assert conn.execute(f"SELECT COUNT(*) FROM {PAYLOAD_TABLE}").fetchone()[0] == 179
        assert conn.execute(f"SELECT COUNT(*) FROM {COVERAGE_TABLE}").fetchone()[0] == 496


def test_all_staging_records_remain_review_required_unverified() -> None:
    with _conn() as conn:
        unsafe = conn.execute(
            """
            SELECT COUNT(*) FROM mkb_review_staging_records
            WHERE imported_by_block=? AND (review_required<>1 OR verified<>0 OR auto_accepted<>0)
            """,
            (R23_IMPORTED_BLOCK,),
        ).fetchone()[0]
        active = conn.execute(
            "SELECT COUNT(*) FROM mkb_review_staging_records WHERE imported_by_block=? AND tier='active'",
            (R23_IMPORTED_BLOCK,),
        ).fetchone()[0]
    assert unsafe == 0
    assert active == 0


def test_detail_viewer_shows_full_schema_and_minimal_payloads() -> None:
    with _conn() as conn:
        full_id = conn.execute(
            "SELECT record_id FROM mkb_review_staging_payloads WHERE package_type='full_schema' LIMIT 1"
        ).fetchone()[0]
        minimal_id = conn.execute(
            "SELECT record_id FROM mkb_review_staging_payloads WHERE package_type='minimal_review_bound' LIMIT 1"
        ).fetchone()[0]
    full = get_staging_detail(full_id)
    minimal = get_staging_detail(minimal_id)
    assert full["payload_available"] is True
    assert full["extracted_sections"]
    assert full["extracted_items"]
    assert full["quality_metrics"]["schema_valid"] is True
    assert minimal["payload_available"] is True
    assert minimal["quality_metrics"]["minimal_review"] is True


def test_detail_viewer_shows_review_only_and_non_sendable_reasons() -> None:
    with _conn() as conn:
        review_id = conn.execute(
            "SELECT staging_id FROM mkb_review_staging_records WHERE package_type='review_only_finalized' LIMIT 1"
        ).fetchone()[0]
        excluded_id = conn.execute(
            "SELECT staging_id FROM mkb_review_staging_records WHERE package_type='non_sendable_excluded' LIMIT 1"
        ).fetchone()[0]
    review = build_staging_quality_view(review_id)
    excluded = build_staging_quality_view(excluded_id)
    assert review["payload_available"] is False
    assert review["terminal_reason"]
    assert excluded["payload_available"] is False
    assert excluded["terminal_reason"] == "excluded_rtf_signal_container"


def test_summary_required_fields_passed() -> None:
    s = _summary()
    assert s["overall_result"] == "PASS"
    assert s["provider_model_call_made"] is False
    assert s["live_extraction_started"] is False
    assert s["r23_staging_records_visible"] == 496
    assert s["content_extracted_records_expected"] == 179
    assert s["payloads_materialized_to_staging"] == 179
    assert s["records_with_payload_available"] == 179
    assert s["records_metadata_only"] == 317
    assert s["source_evidence_refs_created"] == 496
    assert s["quality_viewer_created"] is True
    assert s["source_evidence_viewer_created"] is True
    assert s["corpus_coverage_matrix_created"] is True
    assert s["active_verified_records_created"] == 0
    assert s["auto_accept_enabled"] is False
    assert s["medical_decision_made"] is False
    assert s["privacy_result"] == "passed"
    assert s["safety_result"] == "passed"


def test_public_reports_have_no_private_paths_or_raw_payload_markers() -> None:
    forbidden = ["C:\\", "GEMINI_API_KEY", "Bearer ", "Authorization:", "ya29.", "raw_provider_response"]
    for path in REPORT_DIR.glob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for marker in forbidden:
            assert marker not in text
