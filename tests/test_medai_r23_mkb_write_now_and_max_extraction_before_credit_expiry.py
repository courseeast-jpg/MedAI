"""Tests for R23 MKB write-now staging and max extraction gate."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload
import scripts.run_medai_r23_mkb_write_now_and_max_extraction_before_credit_expiry as mod

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_r23_mkb_write_now_and_max_extraction_before_credit_expiry"


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_local_gate_builds_backup_rollback_and_write_plan() -> None:
    gate = mod.local_gate()
    s = mod.build_summary(gate, mode="local")
    assert s["user_authorized_mkb_write_now"] is True
    assert s["content_packages_before_r23"] == 163
    assert s["write_plan_record_count"] == 480
    assert s["mkb_backup_created"] is True
    assert s["rollback_manifest_created"] is True
    assert s["provider_model_call_made"] is False


def test_live_write_verification_if_present_or_plan_safe() -> None:
    s = _summary()
    assert s["mkb_write_scope"] == "unverified_review_required_only"
    assert s["active_verified_mkb_records_written"] == 0
    assert s["auto_accept_enabled"] is False
    assert s["medical_decision_made"] is False


def test_staging_db_schema_has_review_tables() -> None:
    assert mod.MKB_DB.is_file()
    with sqlite3.connect(str(mod.MKB_DB)) as con:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "mkb_review_staging_records" in tables
    assert "mkb_review_staging_ledger" in tables
    assert "mkb_review_staging_rollback" in tables


def test_written_records_are_review_required_when_live_has_run() -> None:
    s = _summary()
    if not s["mkb_write_completed"]:
        return
    with sqlite3.connect(str(mod.MKB_DB)) as con:
        unsafe = con.execute(
            "SELECT COUNT(*) FROM mkb_review_staging_records WHERE imported_by_block=? AND (review_required<>1 OR verified<>0 OR auto_accepted<>0)",
            (mod.BLOCK,),
        ).fetchone()[0]
        total = con.execute(
            "SELECT COUNT(*) FROM mkb_review_staging_records WHERE imported_by_block=?",
            (mod.BLOCK,),
        ).fetchone()[0]
    assert total >= 480
    assert unsafe == 0


def test_medication_and_truth_safety_flags() -> None:
    s = _summary()
    assert s["medication_safety_gate_applied_or_pending"] is True
    assert s["truth_resolution_applied_or_quarantined"] is True


def test_public_reports_privacy_clean() -> None:
    for report in REPORT_DIR.glob("*"):
        if report.suffix.lower() in {".json", ".md"}:
            text = report.read_text(encoding="utf-8", errors="ignore")
            assert "C:\\Users\\S1" not in text
            assert "MedAI_Private" not in text
            assert "tokenized_content" not in text
            assert "Bearer " not in text
            assert "ya29." not in text
            assert check_public_report_payload(text).passed, report.name


def test_private_artifact_flags_false() -> None:
    s = _summary()
    for key in (
        "private_artifacts_committed", "raw_ai_response_committed", "raw_text_committed",
        "tokenized_payloads_committed", "token_maps_committed", "pi_values_committed",
        "credentials_or_tokens_committed",
    ):
        assert s[key] is False
    assert s["public_report_phi_leak_count"] == 0
    assert s["private_path_leaks_after"] == 0
    assert s["secret_leaks_after"] == 0
