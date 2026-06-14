"""Local-only, no-live tests for the 16E-C repaired-payload machine review."""
from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload

import scripts.run_medai_real_doc_repaired_payload_machine_review_local_only_16e_c as mod

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_real_doc_repaired_payload_machine_review_local_only_16e_c"
DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_REAL_DOC_REPAIRED_PAYLOAD_MACHINE_REVIEW_LOCAL_ONLY_16E_C"

REQUIRED_DOCS = (
    "MEDAI_REAL_DOC_REPAIRED_PAYLOAD_MACHINE_REVIEW_LOCAL_ONLY_16E_C.md",
    "MEDAI_OPERATOR_ATTESTATION_REQUIRED_AFTER_MACHINE_REVIEW_16E_C.md",
    "MEDAI_16D_RETRY_ENTRY_CRITERIA_AFTER_16E_C.md",
    "MEDAI_NEXT_DECISION_16E_C.md",
)


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_public_reports_exist():
    for name in ("summary.json", "implementation_report.md", "machine_review_matrix.md",
                 "public_sanitized_review_assessment.json"):
        assert (REPORT_DIR / name).exists(), name


def test_required_docs_exist():
    for d in REQUIRED_DOCS:
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


def test_no_live_gate_activation_or_path_arg():
    src = inspect.getsource(mod)
    assert "os.environ[" not in src           # gate env only read, never set
    assert "argparse" not in src
    assert "sys.argv" not in src
    assert ".listdir" not in src
    assert "os.walk" not in src


def test_no_raw_repaired_payload_in_repo_reports():
    # Reports must not embed payload body / token maps / raw OCR markers.
    for name in ("summary.json", "implementation_report.md", "machine_review_matrix.md",
                 "public_sanitized_review_assessment.json"):
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        assert "C:\\" not in t
        assert "/home/" not in t
        assert "Labcorp" not in t
        assert '"token_map"' not in t
        assert "Bearer " not in t
        assert "ya29." not in t


def test_summary_safe_fields():
    s = _summary()
    assert s["block"] == "MEDAI-REAL-DOC-REPAIRED-PAYLOAD-MACHINE-REVIEW-LOCAL-ONLY-16E-C"
    assert s["local_only"] is True
    assert s["provider_call_made"] is False
    assert s["vertex_live_execution"] is False
    assert s["billing_api_call_made"] is False
    assert s["future_live_gate_set"] is False
    assert s["future_live_gate_environment_active"] is False
    assert s["sixteen_d_retry_started"] is False
    assert s["repaired_payload_read_locally"] is True
    assert s["raw_labcorp_remaining"] is False
    assert s["tokenized_payload_committed"] is False
    assert s["raw_ocr_committed"] is False
    assert s["token_map_committed"] is False
    assert s["public_report_phi_leak_count"] == 0
    assert s["operator_attestation_still_required"] is True
    assert s["machine_review_result"] in ("PASS", "NEEDS_HUMAN_REVIEW")
    assert s["safety_result"] == "passed"
    # Per spec: NO_PHI_ATTESTED must never be written automatically.
    assert s["machine_review_result"] != "NO_PHI_ATTESTED"
    assert s["privacy_result"] in ("passed", "needs_human_review")


def test_public_assessment_counts_only():
    pa = json.loads((REPORT_DIR / "public_sanitized_review_assessment.json").read_text(encoding="utf-8"))
    assert pa["raw_identifier_leak_count"] == 0
    assert pa["token_map_in_public_report"] is False
    assert pa["operator_attestation_still_required"] is True


def test_public_reports_pass_privacy_check():
    for name in ("summary.json", "implementation_report.md", "machine_review_matrix.md",
                 "public_sanitized_review_assessment.json"):
        r = check_public_report_payload((REPORT_DIR / name).read_text(encoding="utf-8"))
        assert r.raw_phi_logged_in_public_reports is False
        assert r.secret_leaks == 0


def test_no_ssn_pattern_in_reports():
    for name in ("summary.json", "implementation_report.md", "machine_review_matrix.md",
                 "public_sanitized_review_assessment.json"):
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        assert not re.search(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b", t)
