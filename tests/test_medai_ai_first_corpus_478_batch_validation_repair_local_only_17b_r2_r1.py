"""No-live tests for the 17B-R2-R1 478-batch validation repair. Inspect artifacts only."""
from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload

import scripts.run_medai_ai_first_corpus_478_batch_validation_repair_local_only_17b_r2_r1 as mod

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_ai_first_corpus_478_batch_validation_repair_local_only_17b_r2_r1"
DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_AI_FIRST_CORPUS_478_BATCH_VALIDATION_REPAIR_LOCAL_ONLY_17B_R2_R1"

PUBLIC_REPORTS = ("summary.json", "implementation_report.md", "repair_matrix.md",
                  "validation_after_repair_public.json", "dry_run_cost_estimate_public.json",
                  "batch_plan_public.json", "privacy_gate_matrix.md",
                  "live_extraction_entry_gate.md", "no_490_dedupe_check.md")


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_public_reports_exist():
    for name in PUBLIC_REPORTS:
        assert (REPORT_DIR / name).exists(), name


def test_required_docs_exist():
    for d in (
        "MEDAI_AI_FIRST_CORPUS_478_BATCH_VALIDATION_REPAIR_LOCAL_ONLY_17B_R2_R1.md",
        "MEDAI_IDENTIFIER_PATTERN_REPAIR_POLICY_17B_R2_R1.md",
        "MEDAI_478_DOC_ID_DEDUPE_AND_NO_490_BATCH_RULE_17B_R2_R1.md",
        "MEDAI_LIVE_BATCH_ENTRY_CRITERIA_AFTER_17B_R2_R1.md",
        "MEDAI_REMAINING_NON_READY_FILES_EXCLUDED_17B_R2_R1.md",
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
    assert "os.environ[" not in src


def test_summary_no_live_safe_values():
    s = _summary()
    assert s["block"] == "MEDAI-AI-FIRST-CORPUS-478-BATCH-VALIDATION-REPAIR-LOCAL-ONLY-17B-R2-R1"
    assert s["local_only"] is True
    assert s["dry_run_only"] is True
    assert s["provider_call_made"] is False
    assert s["vertex_live_execution"] is False
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
    assert s["ready_files_total"] == 478
    assert s["old_12_added_separately"] is False
    assert s["combined_batch_count"] == 478
    assert s["failed_files_from_17b_r2"] == 42
    assert s["tokenized_payloads_written_to_repo"] is False
    assert s["raw_ocr_written_to_repo"] is False
    assert s["token_maps_written_to_repo"] is False
    assert s["private_identifier_values_written_to_repo"] is False
    assert s["raw_source_files_uploaded"] is False
    assert s["public_report_phi_leak_count"] == 0
    assert s["future_live_extraction_not_started"] is True
    assert s["safety_result"] == "passed"


def test_pass_branch_or_blocked_but_never_called():
    s = _summary()
    if s["privacy_result"] == "passed":
        assert s["files_repaired"] == 42
        assert s["outbound_requests_built"] == 478
        assert s["request_validation_passed"] is True
        assert s["request_validation_failed_count_after_repair"] == 0
    else:
        assert s["privacy_result"] == "blocked"
        assert s["request_validation_passed"] is False
    assert s["provider_call_made"] is False
    assert s["live_extraction_started"] is False


def test_failure_classes_only_known():
    s = _summary()
    assert set(s["failure_classes_from_17b_r2"]).issubset(
        {"accession_specimen", "insurance_account", "mrn", "phone"})


def test_private_outbound_path_outside_repo():
    s = _summary()
    p = Path(s["private_outbound_requests_path"])
    assert REPO_ROOT not in p.parents
    assert "MedAI_Private" in str(p)


def test_public_reports_no_payload_or_secrets():
    for name in PUBLIC_REPORTS:
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        assert "/home/" not in t
        assert "Bearer " not in t
        assert "ya29." not in t
        assert '"tokenized_content"' not in t
    for name in PUBLIC_REPORTS:
        if name == "summary.json":
            continue
        r = check_public_report_payload((REPORT_DIR / name).read_text(encoding="utf-8"))
        assert r.passed, (name, getattr(r, "leak_examples_redacted", None))
    rs = check_public_report_payload((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))
    assert rs.raw_phi_logged_in_public_reports is False
    assert rs.secret_leaks == 0


def test_no_ssn_pattern_in_reports():
    for name in ("summary.json", "validation_after_repair_public.json",
                 "dry_run_cost_estimate_public.json"):
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        assert not re.search(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b", t)
