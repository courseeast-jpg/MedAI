"""No-live tests for the 17B-R1 phone-pattern repair (local-only)."""
from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload

import scripts.run_medai_ai_first_corpus_17b_phone_pattern_repair_local_only_17b_r1 as mod

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_17b_phone_pattern_repair_local_only_17b_r1"
DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_AI_FIRST_CORPUS_17B_PHONE_PATTERN_REPAIR_LOCAL_ONLY_17B_R1"


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_public_reports_exist():
    for name in ("summary.json", "implementation_report.md", "repair_matrix.md",
                 "validation_after_repair_public.json", "dry_run_cost_estimate_public.json",
                 "live_17c_entry_gate.md"):
        assert (REPORT_DIR / name).exists(), name


def test_required_docs_exist():
    for d in (
        "MEDAI_AI_FIRST_CORPUS_17B_PHONE_PATTERN_REPAIR_LOCAL_ONLY_17B_R1.md",
        "MEDAI_PHONE_PATTERN_REPAIR_POLICY_17B_R1.md",
        "MEDAI_17C_ENTRY_CRITERIA_AFTER_17B_R1.md",
        "MEDAI_REMAINING_BLOCKED_CORPUS_FOLLOWUP_17A_R2.md",
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
    assert s["block"] == "MEDAI-AI-FIRST-CORPUS-17B-PHONE-PATTERN-REPAIR-LOCAL-ONLY-17B-R1"
    assert s["local_only"] is True
    assert s["dry_run_only"] is True
    assert s["provider_call_made"] is False
    assert s["vertex_live_execution"] is False
    assert s["gemini_call_made"] is False
    assert s["claude_call_made"] is False
    assert s["openai_call_made"] is False
    assert s["billing_api_call_made"] is False
    assert s["future_live_gate_set"] is False
    assert s["future_live_gate_environment_active"] is False
    assert s["seventeen_c_live_started"] is False
    assert s["mkb_db_opened"] is False
    assert s["active_mkb_write"] is False
    assert s["auto_accept_enabled"] is False
    assert s["medical_decision_made"] is False
    assert s["production_queue_mutated"] is False
    assert s["ready_files_from_17a"] == 12
    assert s["failed_files_from_17b"] == 4
    assert s["failed_reason"] == "phone_pattern"
    assert s["tokenized_payloads_written_to_repo"] is False
    assert s["raw_ocr_written_to_repo"] is False
    assert s["token_maps_written_to_repo"] is False
    assert s["private_identifier_values_written_to_repo"] is False
    assert s["public_report_phi_leak_count"] == 0
    assert s["future_17c_live_not_started"] is True
    assert s["safety_result"] == "passed"


def test_pass_branch_or_blocked_but_never_called():
    s = _summary()
    if s["privacy_result"] == "passed":
        assert s["request_validation_passed"] is True
        assert s["residual_phone_pattern_failures_after_repair"] == 0
        assert s["files_repaired"] == 4
        assert s["outbound_requests_built"] == 12
    else:
        assert s["privacy_result"] == "blocked"
        assert s["request_validation_passed"] is False
    # Never a provider call regardless of branch.
    assert s["provider_call_made"] is False
    assert s["seventeen_c_live_started"] is False


def test_private_outbound_path_outside_repo():
    s = _summary()
    p = Path(s["private_outbound_requests_path"])
    assert REPO_ROOT not in p.parents
    assert "MedAI_Private" in str(p)


def test_public_reports_no_payload_or_secrets():
    for name in ("summary.json", "validation_after_repair_public.json",
                 "dry_run_cost_estimate_public.json", "implementation_report.md",
                 "repair_matrix.md", "live_17c_entry_gate.md"):
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        assert "/home/" not in t
        assert "Bearer " not in t
        assert "ya29." not in t
        assert '"tokenized_content"' not in t
    for name in ("validation_after_repair_public.json", "dry_run_cost_estimate_public.json",
                 "implementation_report.md", "repair_matrix.md", "live_17c_entry_gate.md"):
        r = check_public_report_payload((REPORT_DIR / name).read_text(encoding="utf-8"))
        assert r.passed, (name, getattr(r, "leak_examples_redacted", None))
    rs = check_public_report_payload((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))
    assert rs.raw_phi_logged_in_public_reports is False
    assert rs.secret_leaks == 0


def test_no_ssn_pattern_in_public_reports():
    for name in ("summary.json", "validation_after_repair_public.json",
                 "dry_run_cost_estimate_public.json"):
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        assert not re.search(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b", t)
