"""No-live tests for the 16C-A live-gate-prep package."""
from __future__ import annotations

import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload

import scripts.run_medai_vertex_real_doc_single_pilot_live_gate_prep_no_live_16c_a as mod

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_JSON = {
    "block": "MEDAI-VERTEX-REAL-DOC-SINGLE-PILOT-LIVE-GATE-PREP-NO-LIVE-16C-A",
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
    "active_mkb_write": False,
    "mkb_db_opened": False,
    "auto_accept_enabled": False,
    "medical_decision_made": False,
    "production_queue_mutated": False,
    "future_live_gate_named": True,
    "future_live_gate_set": False,
    "future_live_gate_environment_active": False,
    "operator_approval_required": True,
    "cost_cap_required": True,
    "redaction_preflight_required": True,
    "one_document_limit_required": True,
    "one_call_limit_required": True,
    "stop_on_first_failure_required": True,
    "future_16d_not_started": True,
    "sandbox_treated_as_medai_validation": False,
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
    "no active MKB write",
    "no auto-accept",
    "no medical decision",
    "no billing API call",
    "no production queue mutation",
    "explicit operator approval",
    "cost cap",
    "redaction/tokenization preflight",
    "one-document limit",
    "one-call limit",
    "stop-on-first-failure",
    "rollback",
    "16D is not started",
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _all_docs_norm() -> str:
    return _norm("\n".join(mod._docs().values()))


def test_validator_exits_zero():
    # Run as a subprocess with the gate env explicitly unset.
    env = dict(os.environ)
    env.pop(mod.DEDICATED_GATE_NAME, None)
    proc = subprocess.run(
        [sys.executable, "scripts/run_medai_vertex_real_doc_single_pilot_live_gate_prep_no_live_16c_a.py"],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "16c_a_pass" in proc.stdout


def test_reports_exist():
    report, failures, _ = mod.evaluate()
    mod.write_reports(report, failures)
    assert mod.SUMMARY_JSON.exists()
    assert mod.IMPLEMENTATION_MD.exists()
    assert mod.LIVE_GATE_MATRIX_MD.exists()


def test_all_docs_exist():
    mod.evaluate()
    for path in (
        mod.LIVE_GATE_PREP_MD,
        mod.DEDICATED_GATE_SPEC_MD,
        mod.OPERATOR_APPROVAL_PACKET_MD,
        mod.GATE_ENVIRONMENT_TEMPLATE_MD,
        mod.PRE_16D_READINESS_MATRIX_MD,
        mod.NEXT_DECISION_MD,
    ):
        assert path.exists(), path


def test_json_fields_match_expected_no_live_values():
    report, failures, _ = mod.evaluate()
    assert failures == []
    for key, value in EXPECTED_JSON.items():
        assert report[key] == value, key


def test_dedicated_gate_named_but_not_set():
    docs = _all_docs_norm()
    assert mod.DEDICATED_GATE_NAME in docs
    assert "Current required value: unset or false" in docs
    assert "This block must not set it" in docs
    assert "1, true, or enabled before 16D is a NO-GO" in docs
    report, _, _ = mod.evaluate()
    assert report["future_live_gate_named"] is True
    assert report["future_live_gate_set"] is False


def test_environment_gate_not_active():
    report, _, _ = mod.evaluate()
    assert report["future_live_gate_environment_active"] is False
    # The gate must not be active in the current environment.
    val = os.environ.get(mod.DEDICATED_GATE_NAME)
    assert val is None or val.strip() not in mod.ACTIVE_VALUES


def test_validator_source_makes_no_forbidden_imports():
    src = inspect.getsource(mod)
    for marker in (
        "import google", "from google", "import vertexai", "generativeai",
        "import anthropic", "import openai", "import requests", "import httpx",
        "import aiohttp", "import urllib", "subprocess", "sqlite3", "chromadb",
        "import pypdf", "pytesseract", "pdfminer", "from clinical_knowledge.mkb",
        "cloudbilling", "billing_v1",
    ):
        assert marker not in src, marker
    # The validator must only READ the gate env, never set it.
    assert "os.environ[" not in src
    assert "setenv" not in src


def test_docs_contain_mandatory_phrases():
    docs = _all_docs_norm()
    for phrase in MANDATORY_PHRASES:
        assert _norm(phrase) in docs, phrase


def test_gate_environment_template_sets_nothing():
    mod.evaluate()
    text = _norm(mod.GATE_ENVIRONMENT_TEMPLATE_MD.read_text(encoding="utf-8"))
    assert "contains no command that sets a live gate" in text
    assert "export MEDAI_" not in text
    assert "set MEDAI_" not in text
    assert "$env:MEDAI_" not in text


def test_sandbox_separation_and_16d_not_started():
    docs = _all_docs_norm()
    assert "separate environment evidence only" in docs
    assert "does not prove MedAI real-document readiness" in docs
    assert "16D is not started" in docs


def test_reports_are_public_safe():
    report, failures, _ = mod.evaluate()
    mod.write_reports(report, failures)
    paths = [
        mod.LIVE_GATE_PREP_MD, mod.DEDICATED_GATE_SPEC_MD, mod.OPERATOR_APPROVAL_PACKET_MD,
        mod.GATE_ENVIRONMENT_TEMPLATE_MD, mod.PRE_16D_READINESS_MATRIX_MD, mod.NEXT_DECISION_MD,
        mod.SUMMARY_JSON, mod.IMPLEMENTATION_MD, mod.LIVE_GATE_MATRIX_MD,
    ]
    for path in paths:
        payload = path.read_text(encoding="utf-8")
        assert "C:\\" not in payload
        assert "Bearer " not in payload
        assert "Authorization" + ":" not in payload
        assert "ya29." not in payload
        assert "AIza" not in payload
        assert not re.search(r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b", payload)
        assert check_public_report_payload(payload).passed


def test_summary_json_after_write_has_all_fields():
    report, failures, _ = mod.evaluate()
    mod.write_reports(report, failures)
    summary = json.loads(mod.SUMMARY_JSON.read_text(encoding="utf-8"))
    for key in EXPECTED_JSON:
        assert key in summary, key
