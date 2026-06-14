"""Tests for the 17C-R2-R5 authorized cost-cap update. Inspect artifacts + the live
runner cap constants; never call a provider model."""
from __future__ import annotations

import json
import re
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload
import scripts.run_medai_ai_first_corpus_478_live_batch_vertex_17c_r2 as live

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_17c_r2_authorized_cost_cap_update_17c_r2_r5"
DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_AI_FIRST_CORPUS_17C_R2_AUTHORIZED_COST_CAP_UPDATE_17C_R2_R5"
PUBLIC_REPORTS = ("summary.json", "implementation_report.md", "cost_cap_update_public.json",
                  "live_rerun_entry_gate_public.md", "safety_boundary_public.md")


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_public_reports_exist():
    for name in PUBLIC_REPORTS:
        assert (REPORT_DIR / name).exists(), name


def test_required_docs_exist():
    for d in (
        "MEDAI_17C_R2_AUTHORIZED_COST_CAP_UPDATE_17C_R2_R5.md",
        "MEDAI_17C_R2_COST_CAP_GUARD_POLICY_R5.md",
        "MEDAI_17C_R2_LIVE_RERUN_ENTRY_GATE_AFTER_R5.md",
    ):
        assert (DOC_DIR / d).exists(), d


def test_live_runner_total_cap_is_040():
    assert float(live.CAP_TOTAL) == 0.40


def test_live_runner_per_chunk_cap_and_chunk_size_unchanged():
    assert float(live.CAP_PER_CHUNK) == 0.05
    assert int(live.CHUNK_SIZE) == 25
    assert live.TARGET_MODEL == "gemini-2.5-flash-lite"


def test_estimate_within_new_cap():
    s = _summary()
    assert s["latest_estimated_total_cost_usd"] == 0.325362
    assert s["estimated_total_within_new_cap"] is True
    assert s["latest_estimated_total_cost_usd"] <= s["authorized_hard_cost_cap_total_usd"]


def test_summary_cap_fields():
    s = _summary()
    assert s["block"] == "MEDAI-AI-FIRST-CORPUS-17C-R2-AUTHORIZED-COST-CAP-UPDATE-17C-R2-R5"
    assert s["previous_hard_cost_cap_total_usd"] == 0.25
    assert s["authorized_hard_cost_cap_total_usd"] == 0.40
    assert s["hard_cost_cap_per_chunk_usd"] == 0.05
    assert s["chunk_size"] == 25
    assert s["caps_consistent_in_runner"] is True


def test_summary_safe_no_model_call():
    s = _summary()
    assert s["local_preflight_only"] is True
    assert s["provider_model_call_made"] is False
    assert s["vertex_model_call_made"] is False
    assert s["gemini_call_made"] is False
    assert s["claude_call_made"] is False
    assert s["openai_call_made"] is False
    assert s["billing_api_call_made"] is False
    assert s["live_gate_set"] is False
    assert s["live_extraction_started"] is False
    assert s["mkb_db_opened"] is False
    assert s["active_mkb_write"] is False
    assert s["auto_accept_enabled"] is False
    assert s["medical_decision_made"] is False
    assert s["production_queue_mutated"] is False
    assert s["private_outbound_requests_committed"] is False
    assert s["tokenized_payloads_written_to_repo"] is False
    assert s["raw_ocr_written_to_repo"] is False
    assert s["token_maps_written_to_repo"] is False
    assert s["private_identifier_values_written_to_repo"] is False
    assert s["credential_or_token_written_to_repo"] is False
    assert s["public_report_phi_leak_count"] == 0


def test_ready_flag_consistency():
    s = _summary()
    if s["sealed_batch_valid"] and s["credential_preflight_passed"]:
        assert s["ready_to_rerun_17c_r2_live"] is True
    if not s["credential_preflight_passed"]:
        assert s["ready_to_rerun_17c_r2_live"] is False


def test_no_credentials_or_payload_markers():
    for name in PUBLIC_REPORTS:
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        for m in ("ya29.", "AIza", "-----BEGIN", "refresh_token", "private_key", '"tokenized_content"'):
            assert m not in t, (name, m)


def test_reports_pass_privacy_check():
    for name in PUBLIC_REPORTS:
        r = check_public_report_payload((REPORT_DIR / name).read_text(encoding="utf-8"))
        assert r.raw_phi_logged_in_public_reports is False, name
        assert r.secret_leaks == 0, name


def test_no_ssn_pattern_in_reports():
    for name in PUBLIC_REPORTS:
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        assert not re.search(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b", t)
