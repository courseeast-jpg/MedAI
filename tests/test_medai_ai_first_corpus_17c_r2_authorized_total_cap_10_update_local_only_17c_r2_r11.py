"""Tests for 17C-R2-R11 local-only $10 total cap authorization update."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import scripts.run_medai_ai_first_corpus_17c_r2_authorized_total_cap_10_update_local_only_17c_r2_r11 as mod
import scripts.run_medai_ai_first_corpus_478_live_batch_vertex_17c_r2 as live

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_17c_r2_authorized_total_cap_10_update_local_only_17c_r2_r11"
DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_AI_FIRST_CORPUS_17C_R2_AUTHORIZED_TOTAL_CAP_10_UPDATE_LOCAL_ONLY_17C_R2_R11"


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_script_exits_safely() -> None:
    summary = mod.run()
    assert summary["block"] == mod.BLOCK
    assert summary["provider_model_call_made"] is False
    assert summary["live_extraction_started"] is False


def test_reports_and_docs_exist() -> None:
    mod.run()
    for name in ("summary.json", "implementation_report.md", "cost_cap_update_public.json",
                 "live_entry_gate_public.md", "safety_boundary_public.md"):
        assert (REPORT_DIR / name).exists(), name
    for name in mod.DOC_NAMES:
        assert (DOC_DIR / name).exists(), name


def test_cap_and_output_strategy_values() -> None:
    s = _summary()
    assert s["previous_authorized_hard_cost_cap_total_usd"] == 0.40
    assert s["authorized_hard_cost_cap_total_usd"] == 10.00
    assert s["hard_cost_cap_per_chunk_usd"] == 0.05
    assert s["max_output_tokens"] == 8192
    assert float(live.CAP_TOTAL) == 10.00
    assert float(live.PREVIOUS_CAP_TOTAL) == 0.40
    assert float(live.CAP_PER_CHUNK) == 0.05
    assert live.MAX_OUTPUT_TOKENS == 8192


def test_cost_gates_within_authorized_cap() -> None:
    s = _summary()
    assert s["estimated_total_cost_with_new_output_ceiling_usd"] == 1.228029
    assert s["estimated_total_within_authorized_cap"] is True
    assert s["selected_chunk_size"] == 17
    assert s["estimated_per_chunk_cost_with_selected_chunk_size_usd"] <= 0.05
    assert s["estimated_chunks_within_per_chunk_cap"] is True
    assert s["requires_new_cost_authorization"] is False


def test_preflight_and_preservation_gates() -> None:
    s = _summary()
    assert s["canonical_batch_valid"] is True
    assert s["request_count_total"] == 478
    assert isinstance(s["credential_preflight_passed"], bool)
    assert s["checkpoint_resume_preserved"] is True
    assert s["failed_evidence_preservation_preserved"] is True
    assert s["public_report_redaction_preserved"] is True
    expected_ready = (
        s["canonical_batch_valid"]
        and s["credential_preflight_passed"]
        and s["estimated_total_within_authorized_cap"]
        and s["estimated_chunks_within_per_chunk_cap"]
        and s["checkpoint_resume_preserved"]
        and s["failed_evidence_preservation_preserved"]
        and s["public_report_redaction_preserved"]
    )
    assert s["ready_to_resume_17c_r2_live"] is bool(expected_ready)


def test_no_live_provider_or_mkb_side_effects() -> None:
    s = _summary()
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


def test_no_private_artifacts_or_credentials_in_public_reports() -> None:
    s = _summary()
    assert s["private_responses_committed"] is False
    assert s["parsed_private_responses_committed"] is False
    assert s["tokenized_payloads_written_to_repo"] is False
    assert s["raw_ocr_written_to_repo"] is False
    assert s["token_maps_written_to_repo"] is False
    assert s["private_identifier_values_written_to_repo"] is False
    assert s["credential_or_token_written_to_repo"] is False
    assert s["public_report_phi_leak_count"] == 0
    assert s["private_filename_path_leaks_after"] == 0
    assert s["secret_leaks_after"] == 0
    for path in REPORT_DIR.iterdir():
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        assert "Bearer " not in text
        assert "ya29." not in text
        assert "AIza" not in text
        assert '"tokenized_content"' not in text
        assert "live_responses_private" not in text
        assert "parsed_responses_private" not in text


def test_r11_script_does_not_call_live_runner_or_set_gate() -> None:
    src = inspect.getsource(mod)
    assert "live.run(" not in src
    assert "os.environ[" not in src
    assert "_default_http_post" not in src
