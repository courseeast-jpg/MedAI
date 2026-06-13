"""No-live tests for the 16C-C final preflight package."""
from __future__ import annotations

import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload

import scripts.run_medai_vertex_real_doc_single_pilot_final_preflight_no_live_16c_c as mod

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_JSON = {
    "block": "MEDAI-VERTEX-REAL-DOC-SINGLE-PILOT-FINAL-PREFLIGHT-NO-LIVE-16C-C",
    "no_live": True,
    "provider_call_made": False,
    "vertex_live_execution": False,
    "gemini_live_execution": False,
    "claude_live_execution": False,
    "openai_live_execution": False,
    "billing_api_call_made": False,
    "real_private_document_processed": False,
    "private_corpus_read": False,
    "whole_corpus_processed": False,
    "pdf_or_image_processed": False,
    "ocr_routing_executed": False,
    "mkb_db_opened": False,
    "active_mkb_write": False,
    "auto_accept_enabled": False,
    "medical_decision_made": False,
    "production_queue_mutated": False,
    "future_live_gate_named": True,
    "future_live_gate_set": False,
    "future_live_gate_environment_active": False,
    "sixteen_a_verified": True,
    "sixteen_b_verified": True,
    "sixteen_c_a_verified": True,
    "sixteen_c_b_verified": True,
    "sixteen_c_b_raw_identifier_leak_count": 0,
    "sixteen_c_b_token_map_public_report": False,
    "sixteen_c_b_outbound_payload_tokenized": True,
    "operator_approval_required_before_16d": True,
    "cost_cap_required_before_16d": True,
    "redaction_preflight_required_before_16d": True,
    "one_document_limit_required_before_16d": True,
    "one_call_limit_required_before_16d": True,
    "stop_on_first_failure_required_before_16d": True,
    "rollback_required_before_16d": True,
    "future_16d_not_started": True,
    "sandbox_treated_as_medai_validation": False,
    "privacy_result": "passed",
    "safety_result": "passed",
}

MANDATORY_PHRASES = (
    "no-live",
    "no provider call",
    "no Vertex live execution",
    "no real/private document processing",
    "no private corpus read",
    "no corpus processing",
    "no PDF/image/OCR processing",
    "no billing API call",
    "no MKB write",
    "no auto-accept",
    "no medical decision",
    "no production queue mutation",
    "explicit operator approval",
    "cost cap",
    "redaction/tokenization proof",
    "one-document limit",
    "one-call limit",
    "stop-on-first-failure",
    "rollback",
    "evidence capture",
    "16D is not started",
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _all_docs_lower() -> str:
    return _norm("\n".join(mod._docs().values())).lower()


def test_verifier_exits_zero():
    env = dict(os.environ)
    env.pop(mod.DEDICATED_GATE_NAME, None)
    proc = subprocess.run(
        [sys.executable, "scripts/run_medai_vertex_real_doc_single_pilot_final_preflight_no_live_16c_c.py"],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "16c_c_pass" in proc.stdout


def test_reports_exist():
    report, failures, inventory = mod.evaluate()
    mod.write_reports(report, failures, inventory)
    assert mod.SUMMARY_JSON.exists()
    assert mod.IMPLEMENTATION_MD.exists()
    assert mod.FINAL_GO_NO_GO_MATRIX_MD.exists()
    assert mod.PRIOR_BLOCK_INVENTORY_JSON.exists()


def test_all_docs_exist():
    mod.evaluate()
    for path in (
        mod.FINAL_PREFLIGHT_MD, mod.FINAL_GO_NO_GO_MD, mod.ENTRY_CRITERIA_MD,
        mod.OPERATOR_APPROVAL_MD, mod.COST_PRIVACY_SAFETY_MD, mod.HANDOFF_TEMPLATE_MD,
        mod.NEXT_DECISION_MD,
    ):
        assert path.exists(), path


def test_json_fields_match_expected_no_live_values():
    report, failures, _ = mod.evaluate()
    assert failures == [], failures
    for key, value in EXPECTED_JSON.items():
        assert report[key] == value, key


def test_prior_block_verification_flags_true():
    report, _, _ = mod.evaluate()
    assert report["sixteen_a_verified"] is True
    assert report["sixteen_b_verified"] is True
    assert report["sixteen_c_a_verified"] is True
    assert report["sixteen_c_b_verified"] is True


def test_16c_b_leak_fields_remain_safe():
    report, _, _ = mod.evaluate()
    assert report["sixteen_c_b_raw_identifier_leak_count"] == 0
    assert report["sixteen_c_b_token_map_public_report"] is False
    assert report["sixteen_c_b_outbound_payload_tokenized"] is True


def test_future_live_gate_named_but_not_set_or_active():
    report, _, _ = mod.evaluate()
    assert report["future_live_gate_named"] is True
    assert report["future_live_gate_set"] is False
    assert report["future_live_gate_environment_active"] is False
    val = os.environ.get(mod.DEDICATED_GATE_NAME)
    assert val is None or val.strip() not in mod.ACTIVE_VALUES


def test_validator_source_no_forbidden_imports():
    src = inspect.getsource(mod)
    for marker in (
        "import google", "from google", "import vertexai", "generativeai",
        "import anthropic", "import openai", "import requests", "import httpx",
        "import aiohttp", "import urllib", "subprocess", "sqlite3", "chromadb",
        "import pypdf", "pytesseract", "pdfminer", "from clinical_knowledge.mkb",
        "cloudbilling", "billing_v1",
    ):
        assert marker not in src, marker
    assert "os.environ[" not in src


def test_docs_contain_mandatory_phrases():
    docs = _all_docs_lower()
    for phrase in MANDATORY_PHRASES:
        assert _norm(phrase).lower() in docs, phrase


def test_prior_block_inventory_lists_four_blocks():
    report, failures, inventory = mod.evaluate()
    mod.write_reports(report, failures, inventory)
    inv = json.loads(mod.PRIOR_BLOCK_INVENTORY_JSON.read_text(encoding="utf-8"))
    for key in ("16A", "16B", "16C-A", "16C-B"):
        assert key in inv
        assert inv[key]["exists"] is True
        assert inv[key]["verified"] is True


def test_reports_are_public_safe():
    report, failures, inventory = mod.evaluate()
    mod.write_reports(report, failures, inventory)
    for path in (
        mod.FINAL_PREFLIGHT_MD, mod.FINAL_GO_NO_GO_MD, mod.ENTRY_CRITERIA_MD,
        mod.OPERATOR_APPROVAL_MD, mod.COST_PRIVACY_SAFETY_MD, mod.HANDOFF_TEMPLATE_MD,
        mod.NEXT_DECISION_MD, mod.SUMMARY_JSON, mod.IMPLEMENTATION_MD,
        mod.FINAL_GO_NO_GO_MATRIX_MD, mod.PRIOR_BLOCK_INVENTORY_JSON,
    ):
        payload = path.read_text(encoding="utf-8")
        assert "C:\\" not in payload
        assert "Bearer " not in payload
        assert "Authorization" + ":" not in payload
        assert "ya29." not in payload
        assert "AIza" not in payload
        assert not re.search(r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b", payload)
        assert check_public_report_payload(payload).passed


def test_summary_json_after_write_has_all_fields():
    report, failures, inventory = mod.evaluate()
    mod.write_reports(report, failures, inventory)
    summary = json.loads(mod.SUMMARY_JSON.read_text(encoding="utf-8"))
    for key in EXPECTED_JSON:
        assert key in summary, key
