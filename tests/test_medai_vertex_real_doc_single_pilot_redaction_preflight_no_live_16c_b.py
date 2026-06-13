"""No-live tests for the 16C-B redaction/tokenization preflight package."""
from __future__ import annotations

import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload

import scripts.run_medai_vertex_real_doc_single_pilot_redaction_preflight_no_live_16c_b as mod

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_JSON = {
    "block": "MEDAI-VERTEX-REAL-DOC-SINGLE-PILOT-REDACTION-PREFLIGHT-NO-LIVE-16C-B",
    "no_live": True,
    "synthetic_only": True,
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
    "future_live_gate_set": False,
    "future_live_gate_environment_active": False,
    "redaction_preflight_executed": True,
    "tokenization_preflight_executed": True,
    "synthetic_fixture_count": 6,
    "raw_identifier_leak_count": 0,
    "token_map_written_to_public_report": False,
    "outbound_payload_contains_raw_identifier": False,
    "outbound_payload_tokenized": True,
    "operator_approval_required_before_16d": True,
    "cost_cap_required_before_16d": True,
    "one_document_limit_required_before_16d": True,
    "one_call_limit_required_before_16d": True,
    "stop_on_first_failure_required_before_16d": True,
    "future_16d_not_started": True,
    "sandbox_treated_as_medai_validation": False,
    "privacy_result": "passed",
    "safety_result": "passed",
}

MANDATORY_PHRASES = (
    "no-live",
    "synthetic only",
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
    "token maps must never be included in outbound payload",
    "token maps must never appear in public reports",
    "redaction/tokenization preflight",
    "explicit operator approval",
    "cost cap",
    "one-document limit",
    "one-call limit",
    "stop-on-first-failure",
    "rollback",
    "16D is not started",
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _all_docs_lower() -> str:
    return _norm("\n".join(mod._docs().values())).lower()


def _known_raw_values() -> set[str]:
    raws: set[str] = set()
    for fx in mod.synthetic_fixtures():
        _t, _f, _tmap, rv = mod.redact(fx["text"])
        raws.update(rv)
    return {r for r in raws if r}


def test_script_exits_zero():
    env = dict(os.environ)
    env.pop(mod.DEDICATED_GATE_NAME, None)
    proc = subprocess.run(
        [sys.executable, "scripts/run_medai_vertex_real_doc_single_pilot_redaction_preflight_no_live_16c_b.py"],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "16c_b_pass" in proc.stdout


def test_reports_exist():
    report, failures, preview = mod.evaluate()
    mod.write_reports(report, failures, preview)
    assert mod.SUMMARY_JSON.exists()
    assert mod.IMPLEMENTATION_MD.exists()
    assert mod.REDACTION_MATRIX_MD.exists()
    assert mod.SYNTHETIC_PAYLOAD_PREVIEW_JSON.exists()


def test_all_docs_exist():
    mod.evaluate()
    for path in (
        mod.REDACTION_PREFLIGHT_MD, mod.TOKENIZATION_SPEC_MD, mod.SYNTHETIC_FIXTURE_SPEC_MD,
        mod.OUTBOUND_SAFETY_RULES_MD, mod.TOKEN_VAULT_ISOLATION_MD, mod.PRE_16D_PRIVACY_MATRIX_MD,
        mod.NEXT_DECISION_MD,
    ):
        assert path.exists(), path


def test_json_fields_match_expected_no_live_values():
    report, failures, _ = mod.evaluate()
    assert failures == [], failures
    for key, value in EXPECTED_JSON.items():
        assert report[key] == value, key


def test_synthetic_only_and_preflight_flags():
    report, _, _ = mod.evaluate()
    assert report["synthetic_only"] is True
    assert report["redaction_preflight_executed"] is True
    assert report["tokenization_preflight_executed"] is True


def test_no_raw_identifier_leak_and_tokenized():
    report, _, _ = mod.evaluate()
    assert report["raw_identifier_leak_count"] == 0
    assert report["token_map_written_to_public_report"] is False
    assert report["outbound_payload_contains_raw_identifier"] is False
    assert report["outbound_payload_tokenized"] is True


def test_future_live_gate_not_set_or_active():
    report, _, _ = mod.evaluate()
    assert report["future_live_gate_set"] is False
    assert report["future_live_gate_environment_active"] is False
    val = os.environ.get(mod.DEDICATED_GATE_NAME)
    assert val is None or val.strip() not in mod.ACTIVE_VALUES


def test_synthetic_payload_preview_has_no_raw_identifiers_or_token_map():
    report, failures, preview = mod.evaluate()
    mod.write_reports(report, failures, preview)
    blob = mod.SYNTHETIC_PAYLOAD_PREVIEW_JSON.read_text(encoding="utf-8")
    for raw in _known_raw_values():
        assert raw not in blob, raw
    # No token-map key/value pair present.
    assert '"token_map"' not in blob
    # Preview must contain tokens.
    assert "[PATIENT_NAME_1]" in blob


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
    # Must only READ the gate env, never set it.
    assert "os.environ[" not in src
    # No document-path CLI argument handling.
    assert "argparse" not in src
    assert "sys.argv" not in src


def test_docs_contain_mandatory_phrases():
    docs = _all_docs_lower()
    for phrase in MANDATORY_PHRASES:
        assert _norm(phrase).lower() in docs, phrase


def test_reports_are_public_safe():
    report, failures, preview = mod.evaluate()
    mod.write_reports(report, failures, preview)
    raws = _known_raw_values()
    for path in (
        mod.REDACTION_PREFLIGHT_MD, mod.TOKENIZATION_SPEC_MD, mod.SYNTHETIC_FIXTURE_SPEC_MD,
        mod.OUTBOUND_SAFETY_RULES_MD, mod.TOKEN_VAULT_ISOLATION_MD, mod.PRE_16D_PRIVACY_MATRIX_MD,
        mod.NEXT_DECISION_MD, mod.SUMMARY_JSON, mod.IMPLEMENTATION_MD, mod.REDACTION_MATRIX_MD,
        mod.SYNTHETIC_PAYLOAD_PREVIEW_JSON,
    ):
        payload = path.read_text(encoding="utf-8")
        assert "C:\\" not in payload
        assert "Bearer " not in payload
        assert "Authorization" + ":" not in payload
        assert "ya29." not in payload
        assert "AIza" not in payload
        for raw in raws:
            assert raw not in payload, (path.name, raw)
        assert check_public_report_payload(payload).passed


def test_summary_json_after_write_has_all_fields():
    report, failures, preview = mod.evaluate()
    mod.write_reports(report, failures, preview)
    summary = json.loads(mod.SUMMARY_JSON.read_text(encoding="utf-8"))
    for key in EXPECTED_JSON:
        assert key in summary, key
