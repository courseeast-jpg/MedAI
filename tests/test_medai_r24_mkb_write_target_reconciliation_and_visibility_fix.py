"""Tests for R24 MKB write target reconciliation and Explorer visibility."""
from __future__ import annotations

import json
from pathlib import Path

from app.mkb_explorer_model import build_mkb_explorer_model
import scripts.run_medai_r24_mkb_write_target_reconciliation_and_visibility_fix as r24


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_r24_mkb_write_target_reconciliation_and_visibility_fix"


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_r23_durable_records_are_found_in_review_staging_db() -> None:
    staging = r24._inspect_r23_staging()
    assert staging["write_target_classification"] == "sqlite_db"
    assert staging["durable_records_found"] == 496
    assert staging["unsafe_count"] == 0
    assert staging["package_type_counts"] == {
        "full_schema": 161,
        "minimal_review_bound": 18,
        "non_sendable_excluded": 2,
        "review_only_finalized": 315,
    }


def test_data_mkb_db_is_not_standard_sqlite_in_this_runtime() -> None:
    assert r24.DATA_MKB_DB.is_file()
    assert r24._is_sqlite_file(r24.DATA_MKB_DB) is False


def test_mkb_explorer_model_exposes_r23_staging_without_active_promotion() -> None:
    model = build_mkb_explorer_model(None, include_local_review_staging=True, tier_filter="review_bound")
    counts = model["counts"]
    assert model["available"] is True
    assert counts["active"] == 0
    assert counts["active_verified"] == 0
    assert counts["review_required_staging"] == 496
    assert counts["r23_imported"] == 496
    assert model["local_review_staging"]["all_review_required"] is True
    assert model["rows"]
    assert all(row["requires_review"] is True for row in model["rows"])


def test_r24_summary_required_fields_are_safe_and_passed() -> None:
    summary = _summary()
    assert summary["block"] == r24.BLOCK
    assert summary["overall_result"] == "PASS"
    assert summary["r23_reported_records_written"] == 496
    assert summary["r23_durable_records_found"] == 496
    assert summary["r23_write_target_classification"] == "sqlite_db"
    assert summary["app_mkb_reader_target_matches_r23_target"] is True
    assert summary["reconciliation_insert_performed"] is False
    assert summary["records_inserted_or_made_visible"] == 496
    assert summary["active_verified_records_created"] == 0
    assert summary["all_visible_records_review_required"] is True
    assert summary["mkb_explorer_staging_visibility_added"] is True
    assert summary["mkb_explorer_r23_visible_count"] == 496
    assert summary["idempotency_verified"] is True
    assert summary["rollback_manifest_preserved_or_created"] is True
    assert summary["provider_model_call_made"] is False
    assert summary["live_extraction_started"] is False
    assert summary["auto_accept_enabled"] is False
    assert summary["medical_decision_made"] is False
    assert summary["privacy_result"] == "passed"
    assert summary["safety_result"] == "passed"


def test_public_reports_do_not_leak_private_paths_or_secrets() -> None:
    forbidden = ["C:\\", "GEMINI_API_KEY", "Bearer ", "Authorization:", "ya29.", "raw_provider_response"]
    for path in REPORT_DIR.glob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for needle in forbidden:
            assert needle not in text
