"""Local-only tests for 17A corpus PI tokenization preparation."""
from __future__ import annotations

import inspect
import json
import subprocess
from pathlib import Path

import scripts.run_medai_ai_first_corpus_pi_tokenization_local_only_17a as mod

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_pi_tokenization_local_only_17a"
DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_AI_FIRST_CORPUS_PI_TOKENIZATION_LOCAL_ONLY_17A"


def _run_script() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python", "scripts/run_medai_ai_first_corpus_pi_tokenization_local_only_17a.py"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_script_exits_safely() -> None:
    result = _run_script()
    assert result.returncode == 0
    assert "MEDAI-AI-FIRST-CORPUS-PI-TOKENIZATION-LOCAL-ONLY-17A" in result.stdout
    assert "provider_call_made" in result.stdout


def test_public_reports_and_docs_exist() -> None:
    _run_script()
    for name in (
        "summary.json",
        "implementation_report.md",
        "corpus_inventory_public.json",
        "tokenization_status_public.csv",
        "privacy_gate_matrix.md",
        "ai_extraction_readiness_public.json",
        "postponed_tasks_register.md",
    ):
        assert (REPORT_DIR / name).exists(), name
    for name in (
        "MEDAI_AI_FIRST_CORPUS_PI_TOKENIZATION_LOCAL_ONLY_17A.md",
        "MEDAI_PRIVATE_IDENTIFIER_VAULT_SPEC_17A.md",
        "MEDAI_CLINICAL_PRESERVE_ALLOWLIST_SPEC_17A.md",
        "MEDAI_TOKENIZED_CORPUS_PACKAGE_SPEC_17A.md",
        "MEDAI_AI_EXTRACTION_NEXT_STAGE_17B_HANDOFF.md",
        "MEDAI_POSTPONED_TASKS_DUE_TO_AI_FIRST_PIVOT_17A.md",
    ):
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
    assert "MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_LIVE_APPROVED" not in src
    assert "os.environ[" not in src
    s = _summary()
    assert s["provider_call_made"] is False
    assert s["vertex_live_execution"] is False
    assert s["gemini_call_made"] is False
    assert s["claude_call_made"] is False
    assert s["openai_call_made"] is False
    assert s["billing_api_call_made"] is False
    assert s["future_live_gate_set"] is False
    assert s["future_live_gate_environment_active"] is False
    assert s["mkb_db_opened"] is False
    assert s["active_mkb_write"] is False
    assert s["auto_accept_enabled"] is False
    assert s["medical_decision_made"] is False
    assert s["production_queue_mutated"] is False


def test_corpus_root_and_private_paths_are_hard_bound_outside_repo() -> None:
    s = _summary()
    assert s["corpus_root"] == mod.CORPUS_ROOT.as_posix()
    assert mod.CORPUS_ROOT == Path("G:/Codex/2026-04-22-connect-github/full_corpus_input")
    assert not str(mod.VAULT_CSV).startswith(str(REPO_ROOT))
    assert not str(mod.TOKENIZED_ROOT).startswith(str(REPO_ROOT))
    assert mod.VAULT_CSV.exists()
    assert mod.TOKENIZED_ROOT.exists()


def test_vault_required_when_private_vault_has_no_values() -> None:
    s = _summary()
    if not mod._vault_has_values():
        assert s["vault_status"] == "VAULT_REQUIRED"
        assert s["exact_identifier_vault_used"] is False
        assert s["tokenized_corpus_generated"] is False
        assert s["privacy_result"] == "vault_required"
    assert s["safety_result"] == "passed"


def test_allowlist_exists_and_broad_name_guessing_disabled() -> None:
    s = _summary()
    assert mod.ALLOWLIST_PATH.exists()
    assert s["clinical_preserve_allowlist_created"] is True
    assert s["broad_name_guessing_used"] is False


def test_private_and_raw_artifacts_are_not_written_to_repo_reports() -> None:
    s = _summary()
    assert s["private_identifier_values_written_to_repo"] is False
    assert s["raw_corpus_files_written_to_repo"] is False
    assert s["raw_ocr_written_to_repo"] is False
    assert s["token_maps_written_to_repo"] is False
    assert s["tokenized_corpus_written_to_repo"] is False
    assert s["public_report_phi_leak_count"] == 0
    for name in (
        "summary.json",
        "implementation_report.md",
        "corpus_inventory_public.json",
        "tokenization_status_public.csv",
        "privacy_gate_matrix.md",
        "ai_extraction_readiness_public.json",
        "postponed_tasks_register.md",
    ):
        text = (REPORT_DIR / name).read_text(encoding="utf-8")
        assert '"token_map"' not in text
        assert "token_map_private" not in text
        assert "extracted_text_raw" not in text
        assert "tokenized_text" not in text
        assert "Bearer " not in text
        assert "ya29." not in text
        assert "AIza" not in text


def test_if_vault_active_outputs_are_outside_repo_and_invisible_to_git() -> None:
    s = _summary()
    if s["vault_status"] == "ACTIVE":
        assert s["tokenized_corpus_generated"] is True
        status = subprocess.check_output(["git", "status", "--short", "--", str(mod.TOKENIZED_ROOT)], cwd=REPO_ROOT, text=True)
        assert status.strip() == ""
