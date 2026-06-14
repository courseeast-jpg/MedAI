"""No-live tests for the 17B AI-extraction schema/batch dry run.

Tests inspect only generated public artifacts. They never make a provider call and
never read private tokenized payload bodies into assertions.
"""
from __future__ import annotations

import inspect
import json
import os
import re
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload

import scripts.run_medai_ai_first_corpus_extraction_schema_batch_dry_run_17b as mod

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_extraction_schema_batch_dry_run_17b"
DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_AI_FIRST_CORPUS_EXTRACTION_SCHEMA_BATCH_DRY_RUN_17B"
SCHEMA_PATH = REPO_ROOT / "config" / "medai_ai_extraction_schema_17b.json"
PROMPT_CONTRACT_PATH = REPO_ROOT / "config" / "medai_ai_extraction_prompt_contract_17b.md"


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_public_reports_exist():
    for name in ("summary.json", "implementation_report.md", "ready_batch_public.json",
                 "schema_validation_public.json", "dry_run_cost_estimate_public.json",
                 "privacy_gate_matrix.md", "live_17c_entry_gate.md"):
        assert (REPORT_DIR / name).exists(), name


def test_schema_exists_and_valid_json():
    assert SCHEMA_PATH.exists()
    obj = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert obj["schema_name"] == "medai_ai_extraction_schema_17b"
    assert "extracted_labs" in obj["fields"]
    assert "evidence_quote_tokenized" in obj["objects"]["lab"]
    assert "evidence_quote_tokenized" in obj["objects"]["medication"]


def test_prompt_contract_exists():
    assert PROMPT_CONTRACT_PATH.exists()
    t = PROMPT_CONTRACT_PATH.read_text(encoding="utf-8")
    assert "strict JSON only" in t
    assert "Do not decode" in t


def test_required_docs_exist():
    for d in (
        "MEDAI_AI_FIRST_CORPUS_EXTRACTION_SCHEMA_BATCH_DRY_RUN_17B.md",
        "MEDAI_AI_EXTRACTION_JSON_SCHEMA_17B.md",
        "MEDAI_AI_EXTRACTION_PROMPT_CONTRACT_17B.md",
        "MEDAI_BATCH_DRY_RUN_AND_COST_CONTROL_17B.md",
        "MEDAI_17C_SMALL_LIVE_BATCH_ENTRY_CRITERIA.md",
        "MEDAI_BLOCKED_CORPUS_EXTRACTION_UNAVAILABLE_FOLLOWUP_17A_R2.md",
    ):
        assert (DOC_DIR / d).exists(), d


def test_no_provider_network_billing_imports():
    src = inspect.getsource(mod)
    for marker in (
        "import requests", "import httpx", "import aiohttp", "import urllib",
        "import google", "from google", "import vertexai", "generativeai",
        "import anthropic", "import openai", "generate_content", "cloudbilling",
        "billing_v1", "import sqlite3", "chromadb", "from clinical_knowledge.mkb",
    ):
        assert marker not in src, marker


def test_no_live_gate_activation():
    src = inspect.getsource(mod)
    assert "os.environ[" not in src  # gate env only read, never set


def test_summary_no_live_safe_values():
    s = _summary()
    assert s["block"] == "MEDAI-AI-FIRST-CORPUS-EXTRACTION-SCHEMA-BATCH-DRY-RUN-17B"
    assert s["dry_run_only"] is True
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
    assert s["ready_files_from_17a"] == 12
    assert s["blocked_files_excluded"] == 587
    assert s["tokenized_payloads_written_to_repo"] is False
    assert s["raw_ocr_written_to_repo"] is False
    assert s["token_maps_written_to_repo"] is False
    assert s["private_identifier_values_written_to_repo"] is False
    assert s["public_report_phi_leak_count"] == 0
    assert s["schema_created"] is True
    assert s["prompt_contract_created"] is True
    assert s["future_17c_live_not_started"] is True
    assert s["safety_result"] == "passed"


def test_validation_passed_or_blocked_but_never_called():
    s = _summary()
    # Either validation passed, or the batch is blocked — but never a provider call.
    assert (s["request_validation_passed"] is True) or (s["privacy_result"] == "blocked")
    assert s["provider_call_made"] is False
    assert s["future_17c_live_not_started"] is True


def test_private_outbound_path_is_outside_repo():
    s = _summary()
    p = Path(s["private_outbound_requests_path"])
    assert REPO_ROOT not in p.parents
    assert "MedAI_Private" in str(p)


def test_public_reports_no_tokenized_payload_or_secrets():
    for name in ("summary.json", "ready_batch_public.json", "schema_validation_public.json",
                 "dry_run_cost_estimate_public.json", "implementation_report.md",
                 "privacy_gate_matrix.md", "live_17c_entry_gate.md"):
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        assert "/home/" not in t
        assert "Bearer " not in t
        assert "ya29." not in t
        assert '"tokenized_content"' not in t  # payload body never published
    # Non-summary reports fully pass the privacy checker; summary may reference the
    # mandated private outbound path (a required field) but carries no PHI/secret.
    for name in ("ready_batch_public.json", "schema_validation_public.json",
                 "dry_run_cost_estimate_public.json", "implementation_report.md",
                 "privacy_gate_matrix.md", "live_17c_entry_gate.md"):
        r = check_public_report_payload((REPORT_DIR / name).read_text(encoding="utf-8"))
        assert r.passed, (name, getattr(r, "leak_examples_redacted", None))
    rs = check_public_report_payload((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))
    assert rs.raw_phi_logged_in_public_reports is False
    assert rs.secret_leaks == 0


def test_no_ssn_pattern_in_public_reports():
    for name in ("summary.json", "ready_batch_public.json", "schema_validation_public.json",
                 "dry_run_cost_estimate_public.json"):
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        assert not re.search(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b", t)
