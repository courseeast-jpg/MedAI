"""Tests for R30: readable-view import fix + operator-runtime proof.

Validates the stable import API and the live operator-runtime proof artifacts. Content-free
assertions only (counts, booleans, heading presence). No raw clinical text.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "medai_r30_fix_r29_readable_view_import_and_prove_operator_runtime"
TEXT_REPORTS = ("summary.json", "implementation_report.md", "rendered_ui_text_probe.txt",
                "ui_evidence.json", "r29_invalid_pass_analysis.md")
SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"ya29\."),
    re.compile(r"Bearer\s+[A-Za-z0-9._-]+", re.IGNORECASE),
    re.compile(r"\bMRN\b"),
    re.compile(r"\bDOB\b"),
    re.compile(r"raw_provider_response", re.IGNORECASE),
]


def _summary() -> dict:
    return json.loads((REPORT_DIR / "summary.json").read_text(encoding="utf-8"))


def test_stable_api_import_works():
    # The exact symbol must be importable from the comparator module (stable API).
    from app.mkb_all_records_qa_comparator import readable_record_view  # noqa: F401
    assert callable(readable_record_view)


def test_artifacts_exist():
    for name in TEXT_REPORTS:
        assert (REPORT_DIR / name).exists(), name


def test_overall_pass_and_runtime_import_clean():
    s = _summary()
    assert s["overall_result"] == "PASS"
    assert s["r29_invalid_pass_confirmed"] is True
    assert s["readable_record_view_import_ok_after_fix"] is True
    assert s["app_main_import_ok_after_fix"] is True
    assert s["operator_streamlit_command_tested"] is True
    assert s["operator_streamlit_url"] == "http://localhost:8561"
    assert s["operator_runtime_import_error_present"] is False


def test_operator_runtime_visibility():
    s = _summary()
    assert s["mkb_explorer_visible"] is True
    assert s["all_record_qa_comparator_visible"] is True
    assert s["extracted_payload_queue_visible"] is True
    assert s["not_extracted_failure_queue_visible"] is True
    assert s["extracted_payload_count_visible"] == 179
    assert s["not_extracted_count_visible"] == 317
    assert s["readable_headings_visible"] is True


def test_headings_in_ui_evidence():
    e = json.loads((REPORT_DIR / "ui_evidence.json").read_text(encoding="utf-8"))
    headings = e["runtime_checks"]["headings_visible"]
    for h in ("Extracted content", "Extracted sections", "Extracted items / facts",
              "Source evidence / original preview", "QA decision"):
        assert headings.get(h) is True, h
    assert e["import_smoke"]["readable_record_view"] is True
    assert e["import_smoke"]["app_main"] is True


def test_live_proof_method_not_static_only():
    s = _summary()
    assert s["static_code_only_validation"] is False
    assert s["live_ui_proof_method"] == "playwright"


def test_root_cause_documented():
    text = (REPORT_DIR / "r29_invalid_pass_analysis.md").read_text(encoding="utf-8")
    assert "sys.modules" in text
    assert "restart" in text.lower()
    assert "readable_record_view" in text


def test_safety_and_no_side_effects():
    s = _summary()
    assert s["provider_model_call_made"] is False
    assert s["live_extraction_started"] is False
    assert s["new_extraction_started"] is False
    assert s["active_verified_records_created"] == 0
    assert s["auto_accept_enabled"] is False
    assert s["medical_decision_made"] is False


def test_no_leaks_in_reports():
    s = _summary()
    assert s["public_report_phi_leak_count"] == 0
    assert s["private_path_leaks_after"] == 0
    assert s["secret_leaks_after"] == 0
    for name in TEXT_REPORTS:
        text = (REPORT_DIR / name).read_text(encoding="utf-8", errors="ignore")
        for pat in SECRET_PATTERNS:
            assert not pat.search(text), (name, pat.pattern)


def test_no_ssn_pattern():
    for name in TEXT_REPORTS:
        text = (REPORT_DIR / name).read_text(encoding="utf-8", errors="ignore")
        assert not re.search(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b", text)
