"""Local-only, no-live tests for the 16E-A no-PHI certification assessment.

These tests inspect only the generated PUBLIC artifacts. They never make a provider
call, never open the private artifact directory's raw contents for assertion, and
never reprocess the approved input file.
"""
from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload

import scripts.run_medai_real_doc_no_phi_certification_local_only_16e_a as mod

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_real_doc_no_phi_certification_local_only_16e_a"
DOC_DIR = REPO_ROOT / "docs" / "pilot_design" / "MEDAI_REAL_DOC_NO_PHI_CERTIFICATION_LOCAL_ONLY_16E_A"

REQUIRED_DOCS = (
    "MEDAI_REAL_DOC_NO_PHI_CERTIFICATION_LOCAL_ONLY_16E_A.md",
    "MEDAI_LOCAL_OCR_DEID_REVIEW_RULES_16E_A.md",
    "MEDAI_OPERATOR_NO_PHI_ATTESTATION_TEMPLATE_16E_A.md",
    "MEDAI_16D_RETRY_ENTRY_CRITERIA_AFTER_16E_A.md",
    "MEDAI_NEXT_DECISION_16E_A.md",
)


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_public_reports_exist():
    for name in ("summary.json", "implementation_report.md", "no_phi_detection_matrix.md",
                 "public_sanitized_payload_assessment.json"):
        assert (REPORT_DIR / name).exists(), name


def test_required_docs_exist():
    for d in REQUIRED_DOCS:
        assert (DOC_DIR / d).exists(), d


def test_no_provider_network_billing_imports_in_script():
    src = inspect.getsource(mod)
    for marker in (
        "import requests", "import httpx", "import aiohttp", "import urllib",
        "import google", "from google", "import vertexai", "generativeai",
        "import anthropic", "import openai", "generate_content", "cloudbilling",
        "billing_v1", "import sqlite3", "chromadb", "from clinical_knowledge.mkb",
    ):
        assert marker not in src, marker


def test_approved_path_hard_bound_and_no_path_argument():
    src = inspect.getsource(mod)
    assert mod.APPROVED_INPUT_PATH == r"G:\MEDICAL\Urine\Archive\2024.03.22\2.PNG"
    # No CLI/document path intake, no folder traversal.
    assert "argparse" not in src
    assert "sys.argv" not in src
    assert "os.walk" not in src
    assert "iterdir" not in src
    assert "glob(" not in src
    assert ".listdir" not in src


def test_no_live_gate_activation_in_script():
    src = inspect.getsource(mod)
    # Gate env is only READ, never set.
    assert "os.environ[" not in src
    # No write/assignment to the dedicated gate name.
    assert f'"{mod.DEDICATED_GATE_NAME}"] =' not in src
    assert f"'{mod.DEDICATED_GATE_NAME}'] =" not in src


def test_summary_no_live_and_safety_fields():
    s = _summary()
    assert s["block"] == "MEDAI-REAL-DOC-NO-PHI-CERTIFICATION-LOCAL-ONLY-16E-A"
    assert s["local_only"] is True
    assert s["one_document_only"] is True
    assert s["folder_processed"] is False
    assert s["additional_file_processed"] is False
    assert s["provider_call_made"] is False
    assert s["vertex_live_execution"] is False
    assert s["billing_api_call_made"] is False
    assert s["future_live_gate_set"] is False
    assert s["future_live_gate_environment_active"] is False
    assert s["mkb_db_opened"] is False
    assert s["active_mkb_write"] is False
    assert s["auto_accept_enabled"] is False
    assert s["medical_decision_made"] is False
    assert s["production_queue_mutated"] is False
    assert s["raw_ocr_written_to_repo"] is False
    assert s["token_map_written_to_repo"] is False
    assert s["public_report_phi_leak_count"] == 0
    assert s["operator_review_required_before_live_retry"] is True
    assert s["future_16d_retry_not_started"] is True
    assert s["private_artifacts_written_outside_repo"] is True
    assert s["safety_result"] == "passed"
    assert s["privacy_result"] in ("passed", "needs_human_review")
    assert s["recommendation"] in ("PASS", "FAIL", "NEEDS_HUMAN_REVIEW")


def test_public_assessment_counts_only_no_raw():
    pa = json.loads((REPORT_DIR / "public_sanitized_payload_assessment.json").read_text(encoding="utf-8"))
    assert pa["raw_identifier_leak_count"] == 0
    assert pa["token_map_in_public_report"] is False
    assert isinstance(pa["detection_class_counts"], dict)
    # All counts are integers (never raw values).
    for v in pa["detection_class_counts"].values():
        assert isinstance(v, int)


def test_reports_have_no_token_map_or_credential_tokens():
    for name in ("summary.json", "implementation_report.md", "no_phi_detection_matrix.md",
                 "public_sanitized_payload_assessment.json"):
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        assert "C:\\" not in t
        assert "/home/" not in t
        assert "Bearer " not in t
        assert "ya29." not in t
        assert "AIza" not in t
        # No token-map style key/value JSON object in public reports.
        assert '"token_map"' not in t


def test_public_reports_carry_no_phi_or_secret():
    # The matrix, assessment, and implementation reports must fully pass the privacy check.
    for name in ("implementation_report.md", "no_phi_detection_matrix.md",
                 "public_sanitized_payload_assessment.json"):
        r = check_public_report_payload((REPORT_DIR / name).read_text(encoding="utf-8"))
        assert r.passed, (name, getattr(r, "leak_examples_redacted", None))
    # summary.json may reference the mandated approved path (a required field), but must
    # carry NO raw PHI and NO secret.
    rs = check_public_report_payload((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))
    assert rs.raw_phi_logged_in_public_reports is False
    assert rs.secret_leaks == 0


def test_no_ssn_pattern_in_reports():
    for name in ("summary.json", "implementation_report.md", "no_phi_detection_matrix.md",
                 "public_sanitized_payload_assessment.json"):
        t = (REPORT_DIR / name).read_text(encoding="utf-8")
        assert not re.search(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b", t)
