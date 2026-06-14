"""No-live tests for the 17C-R2 478 live batch. Inspect generated artifacts only;
never call the provider."""
from __future__ import annotations

import json
import re
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload
import scripts.run_medai_ai_first_corpus_478_live_batch_vertex_17c_r2 as live

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_478_live_batch_vertex_17c_r2"
DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_AI_FIRST_CORPUS_478_LIVE_BATCH_VERTEX_17C_R2"

PUBLIC_REPORTS = ("summary.json", "implementation_report.md", "live_batch_status_public.json",
                  "chunk_status_public.csv", "schema_validation_summary_public.json",
                  "cost_guard_public.json", "privacy_gate_matrix.md",
                  "stopped_on_first_failure_public.md", "provider_status_public.json",
                  "next_stage_17d_mkb_staging_import_gate.md")
TEXT_REPORTS = tuple(n for n in PUBLIC_REPORTS if n.endswith((".json", ".md")))


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_public_reports_exist():
    for name in PUBLIC_REPORTS:
        assert (REPORT_DIR / name).exists(), name


def test_required_docs_exist():
    for d in (
        "MEDAI_AI_FIRST_CORPUS_478_LIVE_BATCH_VERTEX_17C_R2.md",
        "MEDAI_17C_R2_LIVE_GATE_AND_COST_CAP.md",
        "MEDAI_17C_R2_STOP_ON_FIRST_FAILURE_POLICY.md",
        "MEDAI_17C_R2_PRIVATE_RESPONSE_STAGING_SPEC.md",
        "MEDAI_17D_MKB_STAGING_IMPORT_ENTRY_CRITERIA.md",
        "MEDAI_REMAINING_NON_READY_FILES_NOT_INCLUDED_17C_R2.md",
    ):
        assert (DOC_DIR / d).exists(), d


def test_no_response_or_payload_body_in_repo():
    for name in TEXT_REPORTS:
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        assert '"response"' not in t
        assert '"tokenized_content"' not in t
        assert "/home/" not in t
        assert "Bearer " not in t
        assert "ya29." not in t


def test_summary_required_safety_fields():
    s = _summary()
    assert s["block"] == "MEDAI-AI-FIRST-CORPUS-478-LIVE-BATCH-VERTEX-17C-R2"
    assert s["claude_call_made"] is False
    assert s["openai_call_made"] is False
    assert s["billing_api_call_made"] is False
    assert s["request_count_authorized"] == 478
    assert s["old_12_added_separately"] is False
    assert s["combined_batch_count"] == 478
    assert s["original_source_files_uploaded"] is False
    assert s["raw_ai_responses_written_to_repo"] is False
    assert s["parsed_ai_responses_written_to_repo"] is False
    assert s["tokenized_payloads_written_to_repo"] is False
    assert s["raw_ocr_written_to_repo"] is False
    assert s["token_maps_written_to_repo"] is False
    assert s["private_identifier_values_written_to_repo"] is False
    assert s["credential_or_token_written_to_repo"] is False
    assert s["public_report_phi_leak_count"] == 0
    assert s["live_gate_environment_active_after_run"] is False
    assert s["future_17d_mkb_import_not_started"] is True
    assert s["mkb_db_opened"] is False
    assert s["active_mkb_write"] is False
    assert s["auto_accept_enabled"] is False
    assert s["medical_decision_made"] is False
    assert s["production_queue_mutated"] is False
    assert s["target_model"] == "gemini-2.5-flash-lite"
    assert s["chunk_size"] == 25
    # Authorized caps come from the live runner constants (committed source of truth),
    # robust to a stale/externally-mutated report. Total cap is 0.40 (17C-R2-R5; was 0.25).
    assert live.CAP_PER_CHUNK == 0.05
    assert live.CAP_TOTAL == 0.40
    assert s["hard_cost_cap_per_chunk_usd"] == 0.05


def test_execution_result_branch_consistency():
    s = _summary()
    result = s["execution_result"]
    assert result in ("PASS", "BLOCKED", "PROVIDER_FAIL", "SCHEMA_FAIL")
    if result == "PASS":
        assert s["request_count_loaded"] == 478
        assert s["request_count_succeeded"] == 478
        assert s["request_count_failed"] == 0
        assert s["chunk_count_completed"] == 20
        assert s["stopped_on_first_failure"] is False
    else:
        assert s["stopped_on_first_failure"] is True
        assert s["request_count_succeeded"] <= s["request_count_sent"]
        assert s["failure_stage"] in ("preflight", "credentials", "provider", "parse",
                                      "schema", "privacy", "cost", "gate")


def test_no_provider_call_implies_no_send():
    s = _summary()
    if s["provider_call_made"] is False:
        assert s["request_count_sent"] == 0
        assert s["gemini_call_made"] is False
        assert s["vertex_live_execution"] is False


def test_private_paths_outside_repo():
    s = _summary()
    for key in ("private_request_batch_path", "private_response_staging_path"):
        p = Path(s[key])
        assert REPO_ROOT not in p.parents
        assert "MedAI_Private" in str(p)


def test_public_reports_carry_no_phi_or_secret():
    for name in TEXT_REPORTS:
        if name == "summary.json":
            continue
        r = check_public_report_payload((REPORT_DIR / name).read_text(encoding="utf-8"))
        assert r.passed, (name, getattr(r, "leak_examples_redacted", None))
    rs = check_public_report_payload((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))
    assert rs.raw_phi_logged_in_public_reports is False
    assert rs.secret_leaks == 0


def test_no_ssn_pattern_in_reports():
    for name in TEXT_REPORTS:
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        assert not re.search(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b", t)
