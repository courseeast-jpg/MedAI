"""No-live tests for the 478-ready 17B-R2 dry-run batch builder."""
from __future__ import annotations

import inspect
import json
import subprocess
from pathlib import Path

import scripts.run_medai_ai_first_corpus_478_ready_batch_dry_run_17b_r2 as mod

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_478_ready_batch_dry_run_17b_r2"
DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_AI_FIRST_CORPUS_478_READY_BATCH_DRY_RUN_17B_R2"


def _run_script() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python", "scripts/run_medai_ai_first_corpus_478_ready_batch_dry_run_17b_r2.py"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_script_exits_safely() -> None:
    result = _run_script()
    assert result.returncode == 0, result.stderr
    assert "MEDAI-AI-FIRST-CORPUS-478-READY-BATCH-DRY-RUN-17B-R2" in result.stdout
    assert "provider_call_made" in result.stdout


def test_public_reports_and_docs_exist() -> None:
    _run_script()
    for name in (
        "summary.json",
        "implementation_report.md",
        "ready_batch_public.json",
        "validation_summary_public.json",
        "failed_validation_summary_public.csv",
        "batch_plan_public.json",
        "dry_run_cost_estimate_public.json",
        "privacy_gate_matrix.md",
        "live_extraction_entry_gate.md",
        "vertex_credentials_blocker_note.md",
    ):
        assert (REPORT_DIR / name).exists(), name
    for name in mod.DOC_NAMES:
        assert (DOC_DIR / name).exists(), name


def test_no_provider_network_billing_or_mkb_imports() -> None:
    src = inspect.getsource(mod)
    for marker in (
        "import requests",
        "import httpx",
        "import aiohttp",
        "import urllib",
        "import google",
        "from google",
        "import vertexai",
        "generativeai",
        "import anthropic",
        "import openai",
        "generate_content",
        "cloudbilling",
        "billing_v1",
        "import sqlite3",
        "chromadb",
        "from clinical_knowledge.mkb",
    ):
        assert marker not in src, marker


def test_no_live_gate_activation_or_mkb_side_effects() -> None:
    src = inspect.getsource(mod)
    assert "os.environ[" not in src
    s = _summary()
    assert s["provider_call_made"] is False
    assert s["vertex_live_execution"] is False
    assert s["gemini_call_made"] is False
    assert s["claude_call_made"] is False
    assert s["openai_call_made"] is False
    assert s["billing_api_call_made"] is False
    assert s["live_gate_set"] is False
    assert s["seventeen_c_rerun_started"] is False
    assert s["mkb_db_opened"] is False
    assert s["active_mkb_write"] is False
    assert s["auto_accept_enabled"] is False
    assert s["medical_decision_made"] is False
    assert s["production_queue_mutated"] is False
    assert s["future_live_extraction_not_started"] is True


def test_summary_contract_and_counts() -> None:
    s = _summary()
    assert s["block"] == mod.BLOCK
    assert s["dry_run_only"] is True
    assert s["local_only"] is True
    assert s["expected_ready_files_from_17a_r2"] == 478
    assert s["actual_ready_files_loaded"] > 0
    assert s["actual_ready_files_loaded"] == 478
    assert s["blocked_files_excluded"] == 121
    assert s["duplicates_excluded"] == 88
    assert s["remaining_extraction_unavailable_excluded"] == 31
    assert s["unsupported_excluded"] == 2
    assert s["raw_source_files_uploaded"] is False
    assert s["tokenized_payloads_written_to_repo"] is False
    assert s["raw_ocr_written_to_repo"] is False
    assert s["token_maps_written_to_repo"] is False
    assert s["private_identifier_values_written_to_repo"] is False
    assert s["public_report_phi_leak_count"] == 0
    assert s["schema_used"] is True
    assert s["prompt_contract_used"] is True
    assert s["estimated_input_tokens"] > 0
    assert s["estimated_output_tokens"] >= 0
    assert s["estimated_cost_usd"] >= 0
    assert s["target_model_for_future_live"] == "gemini-2.5-flash-lite"


def test_validation_branch_contract() -> None:
    s = _summary()
    if s["request_validation_passed"] is True:
        assert s["request_validation_failed_count"] == 0
        assert s["privacy_result"] == "passed"
        assert s["outbound_requests_built"] == s["actual_ready_files_loaded"]
    else:
        assert s["privacy_result"] == "blocked"
        assert s["request_validation_failed_count"] > 0
        assert s["validation_failure_classes"]
        assert s["outbound_requests_built"] < s["actual_ready_files_loaded"]
    assert s["provider_call_made"] is False
    assert s["future_live_extraction_not_started"] is True


def test_private_outbound_path_is_outside_repo_and_invisible_to_git() -> None:
    s = _summary()
    assert "MedAI_Private" in s["private_outbound_requests_path"]
    assert REPO_ROOT not in mod.PRIVATE_OUT.parents
    result = subprocess.run(
        ["git", "status", "--short", "--", str(mod.PRIVATE_OUT)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "outside repository" in result.stderr


def test_public_reports_do_not_contain_payloads_or_secrets() -> None:
    for path in REPORT_DIR.iterdir():
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        assert '"tokenized_content"' not in text
        assert '"prompt_contract"' not in text
        assert "tokenized_text.txt" not in text
        assert "token_map_private" not in text
        assert "extracted_text_raw" not in text
        assert "Bearer " not in text
        assert "ya29." not in text
        assert "AIza" not in text
        assert "full_corpus_input" not in text
