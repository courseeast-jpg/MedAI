"""Focused tests for the 05 one-doc operator validation block."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ONE_DOC_SCRIPT = REPO_ROOT / "scripts" / "run_medai_local_one_doc_operator_validation_05.py"
ONE_DOC_REPORT_DIR = REPO_ROOT / "reports" / "medai_local_one_doc_operator_validation_05"
ONE_DOC_JSON = ONE_DOC_REPORT_DIR / "medai_local_one_doc_operator_validation_05_report.json"
ONE_DOC_MD = ONE_DOC_REPORT_DIR / "medai_local_one_doc_operator_validation_05_report.md"
ONE_DOC_SHORT_MD = ONE_DOC_REPORT_DIR / "MEDAI_LOCAL_ONE_DOC_OPERATOR_VALIDATION_05.md"

SMOKE_SCRIPT = REPO_ROOT / "scripts" / "run_medai_streamlit_launch_smoke_05.py"
SMOKE_REPORT_DIR = REPO_ROOT / "reports" / "medai_streamlit_launch_smoke_05"
SMOKE_JSON = SMOKE_REPORT_DIR / "medai_streamlit_launch_smoke_05_report.json"
SMOKE_MD = SMOKE_REPORT_DIR / "medai_streamlit_launch_smoke_05_report.md"
SMOKE_SHORT_MD = SMOKE_REPORT_DIR / "MEDAI_STREAMLIT_LAUNCH_SMOKE_05.md"


def _run_script(path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(path)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={
            "MEDAI_ALLOW_EXTERNAL_API": "false",
            "MEDAI_LOCAL_ONLY": "true",
            "PATH": os.environ.get("PATH", ""),
        },
    )


@pytest.fixture(scope="module")
def one_doc_payload() -> dict:
    proc = _run_script(ONE_DOC_SCRIPT)
    # The script must always exit 0 when there is no input file (clean exit).
    # When an input file IS present, exit 0 still indicates ready/all_pass.
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads(ONE_DOC_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def smoke_payload() -> dict:
    proc = _run_script(SMOKE_SCRIPT)
    # The helper-import-only path must pass even without Streamlit present.
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads(SMOKE_JSON.read_text(encoding="utf-8"))


def test_one_doc_script_handles_missing_input_cleanly(one_doc_payload):
    # In the sandbox the conclusion is no_input_file; in environments with a
    # real PDF/TXT in test_input/ the conclusion will be one_doc_operator_validation_ready.
    assert one_doc_payload["conclusion"] in {
        "no_input_file",
        "one_doc_operator_validation_ready",
    }
    assert one_doc_payload["external_api_used"] is False
    assert one_doc_payload["auto_accept_enabled"] is False


def test_one_doc_no_input_path_reports_operator_next_step(one_doc_payload):
    if one_doc_payload["conclusion"] != "no_input_file":
        pytest.skip("input file is present; next-step text not applicable")
    f = one_doc_payload["audit_findings"]
    assert f["one_doc_input_present"] is False
    assert "test_input/" in f["operator_next_step"]
    assert f["structured_facts_extracted_count"] == 0
    assert f["review_bound_records_persisted_count"] == 0
    assert f["retrieval_proof_count"] == 0


def test_one_doc_real_record_action_not_executed(one_doc_payload):
    f = one_doc_payload["audit_findings"]
    if one_doc_payload["conclusion"] == "no_input_file":
        # The "no input file" path never reaches the action layer.
        return
    assert f["operator_action_dry_run_executed_on_real_record"] is False


def test_one_doc_privacy_check_passed(one_doc_payload):
    assert one_doc_payload["privacy_check_passed"] is True
    assert one_doc_payload.get("privacy_check_leak_examples_redacted") in (None, [])


def test_one_doc_reports_pass_public_report_privacy_check():
    from clinical_knowledge.privacy import check_public_report_payload

    for path in (ONE_DOC_JSON, ONE_DOC_MD, ONE_DOC_SHORT_MD):
        assert path.is_file(), path
        content = path.read_text(encoding="utf-8")
        target = json.loads(content) if path.suffix == ".json" else content
        result = check_public_report_payload(target)
        assert result.passed, f"{path.name}: {result.leak_examples_redacted}"


def test_smoke_helper_import_layer_always_passes(smoke_payload):
    f = smoke_payload["audit_findings"]
    assert f["helper_import_passed"] is True
    for item in f["helper_import_results"]:
        assert item["passed"] is True, item["module"]


def test_smoke_app_main_import_reports_clearly(smoke_payload):
    f = smoke_payload["audit_findings"]
    smoke = f["app_main_import_smoke"]
    if f["streamlit_present"]:
        assert smoke.get("ran") is True
        assert smoke.get("passed") in (True, False)
    else:
        assert smoke.get("ran") is False
        assert smoke.get("reason")


def test_smoke_brief_launch_does_not_run_without_explicit_flag(smoke_payload):
    f = smoke_payload["audit_findings"]
    launch = f["streamlit_brief_launch"]
    # The default-off behavior: the brief launch never runs unless the
    # MEDAI_STREAMLIT_LAUNCH_SMOKE env var is explicitly set.
    assert launch.get("ran") is False or os.environ.get("MEDAI_STREAMLIT_LAUNCH_SMOKE")


def test_smoke_privacy_check_passed(smoke_payload):
    assert smoke_payload["privacy_check_passed"] is True


def test_smoke_no_external_api_no_auto_accept(smoke_payload):
    assert smoke_payload["external_api_used"] is False
    assert smoke_payload["auto_accept_enabled"] is False


def test_smoke_reports_pass_public_report_privacy_check():
    from clinical_knowledge.privacy import check_public_report_payload

    for path in (SMOKE_JSON, SMOKE_MD, SMOKE_SHORT_MD):
        assert path.is_file(), path
        content = path.read_text(encoding="utf-8")
        target = json.loads(content) if path.suffix == ".json" else content
        result = check_public_report_payload(target)
        assert result.passed, f"{path.name}: {result.leak_examples_redacted}"


def test_powershell_install_helper_present():
    helper = REPO_ROOT / "scripts" / "local_runtime_prepare_minimum_05.ps1"
    assert helper.is_file(), helper
    body = helper.read_text(encoding="utf-8")
    # Sanity: only minimal packages, no cloud SDKs.
    assert "streamlit" in body
    assert "spacy" in body
    assert "chromadb" in body
    assert "PyPDF2" in body
    assert "pytesseract" in body
    for forbidden in ("google-generativeai", "anthropic", "openai", "azure-"):
        assert forbidden not in body, forbidden
    # Must not touch .env
    assert ".env" in body  # text mentions it
    assert "will NOT modify" in body


def test_existing_chain_tests_still_pass():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_extracted_medical_facts.py",
            "tests/test_run_review_extracted_information_preview.py",
            "tests/test_medai_corpus_extraction_to_mkb_minimum_01.py",
            "tests/test_medai_corpus_extraction_to_mkb_minimum_02_pipeline_smoke.py",
            "tests/test_operator_review_actions.py",
            "tests/test_medai_corpus_extraction_to_mkb_minimum_03_operator_review_ux.py",
            "tests/test_medai_corpus_extraction_to_mkb_minimum_04_real_local_operator_validation.py",
            "-q",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
