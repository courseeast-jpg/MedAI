"""Local-only tests for 17A-R2 extraction_unavailable repair."""
from __future__ import annotations

import inspect
import json
import subprocess
from pathlib import Path

import scripts.run_medai_ai_first_corpus_extraction_unavailable_local_repair_17a_r2 as mod

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_extraction_unavailable_local_repair_17a_r2"
DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_AI_FIRST_CORPUS_EXTRACTION_UNAVAILABLE_LOCAL_REPAIR_17A_R2"


def _run_script() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python", "scripts/run_medai_ai_first_corpus_extraction_unavailable_local_repair_17a_r2.py"],
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
    assert "MEDAI-AI-FIRST-CORPUS-EXTRACTION-UNAVAILABLE-LOCAL-REPAIR-17A-R2_PASS" in result.stdout
    assert "provider_call_made" in result.stdout


def test_public_reports_and_docs_exist() -> None:
    _run_script()
    for name in (
        "summary.json",
        "implementation_report.md",
        "extraction_unavailable_triage_public.json",
        "local_extractor_availability_public.json",
        "readiness_delta_public.json",
        "remaining_blockers_public.csv",
        "privacy_gate_matrix.md",
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
    assert s["seventeen_c_live_started"] is False
    assert s["mkb_db_opened"] is False
    assert s["active_mkb_write"] is False
    assert s["auto_accept_enabled"] is False
    assert s["medical_decision_made"] is False
    assert s["production_queue_mutated"] is False


def test_summary_counts_are_safe_and_non_regressing() -> None:
    s = _summary()
    assert s["block"] == mod.BLOCK
    assert s["local_only"] is True
    assert s["initial_total_files_seen"] == 599
    assert s["initial_ready_for_ai_extraction"] == 12
    assert s["initial_extraction_unavailable"] == 497
    assert s["initial_duplicates"] == 88
    assert s["initial_unsupported"] == 2
    assert s["files_attempted_for_local_repair"] >= 0
    assert s["final_ready_for_ai_extraction"] >= s["initial_ready_for_ai_extraction"]
    assert s["final_blocked_for_ai_extraction"] + s["final_ready_for_ai_extraction"] == 599
    assert s["remaining_extraction_unavailable"] <= s["initial_extraction_unavailable"]
    assert s["public_report_phi_leak_count"] == 0
    assert s["privacy_result"] == "passed"
    assert s["safety_result"] == "passed"


def test_private_and_raw_artifacts_are_not_written_to_repo_reports() -> None:
    s = _summary()
    assert s["raw_ocr_written_to_repo"] is False
    assert s["token_maps_written_to_repo"] is False
    assert s["tokenized_corpus_written_to_repo"] is False
    assert s["private_identifier_values_written_to_repo"] is False
    for path in REPORT_DIR.iterdir():
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        assert '"token_map"' not in text
        assert "token_map_private" not in text
        assert "extracted_text_raw" not in text
        assert "tokenized_text" not in text
        assert "Bearer " not in text
        assert "ya29." not in text
        assert "AIza" not in text
        assert "C:\\Users\\S1\\" not in text


def test_private_outputs_are_outside_repo_and_invisible_to_git() -> None:
    assert REPO_ROOT not in mod.base17a.TOKENIZED_ROOT.parents
    assert REPO_ROOT not in mod.base17a.VAULT_CSV.parents
    for private_path in (mod.base17a.TOKENIZED_ROOT, mod.base17a.VAULT_CSV):
        result = subprocess.run(
            ["git", "status", "--short", "--", str(private_path)],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0
        assert "outside repository" in result.stderr


def test_public_remaining_blockers_are_hashed_only() -> None:
    text = (REPORT_DIR / "remaining_blockers_public.csv").read_text(encoding="utf-8")
    assert "document_id,extension,status,extraction_method,residual_high_confidence_pi_pattern_count" in text
    assert "G:/Codex/" not in text
    assert "full_corpus_input" not in text
