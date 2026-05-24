"""Focused tests for MEDAI-CORPUS-EXTRACTION-TO-MKB-MINIMUM-04.

Validation-only block. The test runs the same validation script and
verifies the conclusion is one of the acceptable values and that no
external API, no auto-accept, no PHI leak occurs.
"""
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

REPORT_DIR = REPO_ROOT / "reports" / "medai_corpus_extraction_to_mkb_minimum_04_real_local_operator_validation"
JSON_REPORT_PATH = REPORT_DIR / "medai_corpus_extraction_to_mkb_minimum_04_real_local_operator_validation_report.json"
MD_REPORT_PATH = REPORT_DIR / "medai_corpus_extraction_to_mkb_minimum_04_real_local_operator_validation_report.md"
SHORT_MD_PATH = REPORT_DIR / "MEDAI_CORPUS_EXTRACTION_TO_MKB_MINIMUM_04_REAL_LOCAL_OPERATOR_VALIDATION.md"


@pytest.fixture(scope="module")
def validation_payload() -> dict:
    proc = subprocess.run(
        [
            sys.executable,
            str(
                REPO_ROOT
                / "scripts"
                / "run_medai_corpus_extraction_to_mkb_minimum_04_real_local_operator_validation.py"
            ),
        ],
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
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads(JSON_REPORT_PATH.read_text(encoding="utf-8"))


def test_validation_conclusion_is_one_of_acceptable_states(validation_payload):
    assert validation_payload["conclusion"] in {
        "real_local_operator_validation_ready",
        "private_corpus_not_present_synthetic_ready",
    }
    assert validation_payload["all_pass"] is True


def test_synthetic_chain_re_run_passed(validation_payload):
    f = validation_payload["audit_findings"]
    assert f["synthetic_validation_passed"] is True
    for script, item in f["prior_validation_results"].items():
        assert item["ran"] is True, script
        assert item["all_pass"] is True, script


def test_operator_action_synthetic_proof_counts_all_three(validation_payload):
    f = validation_payload["audit_findings"]
    assert f["operator_action_synthetic_proof_count"] == 3
    syn = f["operator_action_synthetic"]
    assert syn["accept_proof"] is True
    assert syn["reject_proof"] is True
    assert syn["defer_proof"] is True


def test_ui_import_smoke_passed(validation_payload):
    f = validation_payload["audit_findings"]
    assert f["ui_import_smoke_passed"] is True
    smoke = f["ui_import_smoke"]
    for module_name in (
        "app.extracted_information_preview",
        "app.operator_review_actions",
        "execution.extracted_medical_facts",
    ):
        assert smoke["results"].get(module_name) is True, module_name


def test_external_api_and_auto_accept_remain_disabled(validation_payload):
    assert validation_payload["external_api_used"] is False
    assert validation_payload["auto_accept_enabled"] is False
    f = validation_payload["audit_findings"]
    assert f["external_api_used"] is False
    assert f["auto_accept_enabled"] is False


def test_no_raw_text_or_phi_flags_set(validation_payload):
    f = validation_payload["audit_findings"]
    for key in (
        "real_pdf_committed",
        "real_screenshot_committed",
        "raw_text_printed",
        "raw_ocr_text_printed",
        "raw_filenames_printed",
        "private_paths_printed",
        "phi_printed",
    ):
        assert f[key] is False, key


def test_privacy_check_passed(validation_payload):
    assert validation_payload["privacy_check_passed"] is True
    assert validation_payload.get("privacy_check_leak_examples_redacted") in (None, [])


def test_private_corpus_handling_is_safe(validation_payload):
    f = validation_payload["audit_findings"]
    # Either no corpus is present, or counts-only summary was emitted.
    if not f["private_corpus_present"]:
        assert f["private_docs_evaluated_count"] == 0
        assert f["private_doc_handles"] == []
    else:
        # If a private corpus IS present, evaluation is bounded and
        # handles are non-identifying.
        assert f["private_docs_evaluated_count"] <= 3
        for handle in f["private_doc_handles"]:
            assert handle.startswith("localfile_")


def test_streamlit_launch_smoke_reports_clearly(validation_payload):
    f = validation_payload["audit_findings"]
    smoke = f["streamlit_launch_smoke"]
    # Either Streamlit is present and the smoke ran, or it is absent
    # and the script reports a clear reason. Never silently pass.
    if smoke.get("ran"):
        assert smoke.get("passed") in (True, False)
    else:
        assert smoke.get("reason")


def test_practical_mvp_flag_matches_evidence(validation_payload):
    f = validation_payload["audit_findings"]
    expected = (
        f["synthetic_validation_passed"]
        and f["ui_import_smoke_passed"]
        and f["operator_action_synthetic_proof_count"] == 3
    )
    assert f["practical_mvp_ready_for_local_operator_use"] is expected


def test_reports_pass_public_report_privacy_check():
    from clinical_knowledge.privacy import check_public_report_payload

    for path in (JSON_REPORT_PATH, MD_REPORT_PATH, SHORT_MD_PATH):
        assert path.is_file(), path
        content = path.read_text(encoding="utf-8")
        target = json.loads(content) if path.suffix == ".json" else content
        result = check_public_report_payload(target)
        assert result.passed, f"{path.name}: {result.leak_examples_redacted}"


def test_existing_01_02_03_block_tests_still_pass():
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
            "-q",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
