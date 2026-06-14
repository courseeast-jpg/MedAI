"""Tests for 17C-R2-R10 max-output strategy + adaptive cost/chunk planning. Inspect
artifacts and exercise the planner; never call a provider model."""
from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload
from execution import cost_chunk_planner as planner
import scripts.run_medai_ai_first_corpus_17c_r2_max_output_strategy_local_only_17c_r2_r10 as mod
import scripts.run_medai_ai_first_corpus_478_live_batch_vertex_17c_r2 as live

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_17c_r2_max_output_strategy_local_only_17c_r2_r10"
DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_AI_FIRST_CORPUS_17C_R2_MAX_OUTPUT_STRATEGY_LOCAL_ONLY_17C_R2_R10"
PUBLIC_REPORTS = ("summary.json", "implementation_report.md", "max_output_strategy_public.json",
                  "adaptive_cost_plan_public.json", "compact_prompt_contract_public.md",
                  "live_entry_gate_public.md", "safety_boundary_public.md")
_WIN_PATH = re.compile(r"[A-Za-z]:\\")


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_public_reports_exist():
    for name in PUBLIC_REPORTS:
        assert (REPORT_DIR / name).exists(), name


def test_required_docs_exist():
    for d in (
        "MEDAI_17C_R2_MAX_OUTPUT_STRATEGY_LOCAL_ONLY_17C_R2_R10.md",
        "MEDAI_17C_R2_COMPACT_JSON_OUTPUT_POLICY_R10.md",
        "MEDAI_17C_R2_ADAPTIVE_CHUNK_COST_POLICY_R10.md",
        "MEDAI_17C_R2_LIVE_ENTRY_GATE_AFTER_R10.md",
    ):
        assert (DOC_DIR / d).exists(), d


def test_no_provider_no_live_no_mkb():
    s = _summary()
    for k in ("provider_model_call_made", "vertex_model_call_made", "gemini_call_made",
              "claude_call_made", "openai_call_made", "billing_api_call_made",
              "live_gate_set", "live_extraction_started", "mkb_db_opened", "active_mkb_write",
              "auto_accept_enabled", "medical_decision_made", "production_queue_mutated"):
        assert s[k] is False, k
    src = inspect.getsource(mod)
    assert "_default_http_post" not in src
    assert "generate_content" not in src


def test_max_output_tokens_raised_above_2048():
    s = _summary()
    assert s["previous_max_output_tokens"] == 2048
    assert s["new_max_output_tokens"] > 2048
    assert s["max_output_tokens_raised"] is True
    # Live runner constant actually reflects the raised ceiling.
    assert live.MAX_OUTPUT_TOKENS > 2048
    assert live.PREVIOUS_MAX_OUTPUT_TOKENS == 2048


def test_compact_prompt_rules_present():
    s = _summary()
    assert s["compact_output_prompt_hardened"] is True
    contract = live.PROMPT_CONTRACT_PATH.read_text(encoding="utf-8")
    flat = re.sub(r"\s+", " ", contract)
    for marker in ("Compact Output", "No explanatory text", "No duplicate evidence",
                   "Evidence anchors must be SHORT", "Use empty arrays"):
        assert re.sub(r"\s+", " ", marker) in flat, marker


def test_schema_not_weakened():
    s = _summary()
    assert s["schema_weakened"] is False
    assert s["required_schema_keys_preserved"] is True
    assert s["strict_json_preserved"] is True
    # Core required keys unchanged.
    assert "extracted_labs" in live.EXPECTED_TOP_LEVEL_FIELDS
    assert "needs_review" in live.EXPECTED_TOP_LEVEL_FIELDS


def test_checkpoint_and_evidence_preserved():
    s = _summary()
    assert s["checkpoint_resume_preserved"] is True
    assert s["failed_evidence_preservation_preserved"] is True
    src = inspect.getsource(live)
    assert "lc.decide_resume(" in src
    assert "lc.mark_completed(" in src
    assert "lc.mark_failed(" in src
    assert "lc.preserve_failed_evidence(" in src


def test_caps_unchanged():
    s = _summary()
    assert s["authorized_hard_cost_cap_total_usd"] == 0.40
    assert s["hard_cost_cap_per_chunk_usd"] == 0.05
    assert float(live.CAP_TOTAL) == 0.40
    assert float(live.CAP_PER_CHUNK) == 0.05


def test_selected_chunk_size_keeps_chunk_within_cap_or_blocks():
    s = _summary()
    sel = s["selected_chunk_size"]
    per_chunk = s["estimated_per_chunk_cost_with_selected_chunk_size_usd"]
    if s["estimated_chunks_within_per_chunk_cap"]:
        assert sel >= 1
        assert per_chunk is not None and per_chunk <= 0.05
    else:
        # Not within cap -> must not be marked ready for live.
        assert s["ready_to_resume_17c_r2_live"] is False


def test_adaptive_planner_behaviour():
    # At the raised ceiling, chunk 25 exceeds the per-chunk cap and the planner reduces it.
    per_doc = [800] * 30
    sel = planner.select_chunk_size(per_doc, 8192, 0.075, 0.30, 0.05, 25)
    assert 0 < sel < 25
    assert planner.worst_case_chunk_cost(per_doc, 8192, 0.075, 0.30, sel) <= 0.05
    # A tiny ceiling keeps the full chunk size.
    assert planner.select_chunk_size(per_doc, 256, 0.075, 0.30, 0.05, 25) == 25


def test_ready_gate_consistency():
    s = _summary()
    expected_ready = (s["canonical_batch_valid"] and s["credential_preflight_passed"]
                      and s["estimated_total_within_authorized_cap"]
                      and s["estimated_chunks_within_per_chunk_cap"])
    assert s["ready_to_resume_17c_r2_live"] is bool(expected_ready)
    if not s["ready_to_resume_17c_r2_live"] and not s["estimated_total_within_authorized_cap"] \
            and s["estimated_chunks_within_per_chunk_cap"] and s["canonical_batch_valid"]:
        assert s["requires_new_cost_authorization"] is True


def test_no_private_windows_path_in_reports():
    for name in PUBLIC_REPORTS:
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        assert "MedAI_Private" not in t
        assert not _WIN_PATH.search(t), name


def test_no_secret_like_or_raw_ai_response_in_reports():
    for name in PUBLIC_REPORTS:
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        for m in ("ya29.", "AIza", "-----BEGIN", "refresh_token", "private_key", "Bearer ",
                  '"response"', '"candidates"', '"tokenized_content"'):
            assert m not in t, (name, m)


def test_reports_pass_privacy_check():
    s = _summary()
    assert s["public_report_phi_leak_count"] == 0
    assert s["private_filename_path_leaks_after"] == 0
    assert s["secret_leaks_after"] == 0
    for name in PUBLIC_REPORTS:
        r = check_public_report_payload((REPORT_DIR / name).read_text(encoding="utf-8"))
        assert r.passed, (name, getattr(r, "leak_examples_redacted", None))


def test_no_committed_private_flags():
    s = _summary()
    for k in ("private_responses_committed", "parsed_private_responses_committed",
              "tokenized_payloads_written_to_repo", "raw_ocr_written_to_repo",
              "token_maps_written_to_repo", "private_identifier_values_written_to_repo",
              "credential_or_token_written_to_repo"):
        assert s[k] is False, k


def test_no_ssn_pattern_in_reports():
    for name in PUBLIC_REPORTS:
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        assert not re.search(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b", t)
