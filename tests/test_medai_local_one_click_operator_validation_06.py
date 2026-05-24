"""Focused tests for the 06 one-click operator validation block.

Windows PowerShell file-picker UI is not exercised here. The aggregator
and synthetic fallback report path are the testable surfaces.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PS_ORCHESTRATOR = REPO_ROOT / "scripts" / "run_medai_local_one_click_operator_validation_06.ps1"
REPORT_HELPER = REPO_ROOT / "scripts" / "run_medai_local_one_click_operator_validation_06_report.py"
REPORT_DIR = REPO_ROOT / "reports" / "medai_local_one_click_operator_validation_06"
JSON_REPORT_PATH = REPORT_DIR / "medai_local_one_click_operator_validation_06_report.json"
MD_REPORT_PATH = REPORT_DIR / "medai_local_one_click_operator_validation_06_report.md"
SHORT_MD_PATH = REPORT_DIR / "MEDAI_LOCAL_ONE_CLICK_OPERATOR_VALIDATION_06.md"


from scripts.run_medai_local_one_click_operator_validation_06_report import (  # noqa: E402
    ALLOWED_INPUT_MODES,
    ALLOWED_INPUT_SUFFIXES,
    NEXT_UI_COMMAND,
    SAFE_HASH_PREFIX,
    build_report,
    safe_input_hash,
    write_reports,
)


def _expected_safe_hash(size: int, suffix: str) -> str:
    digest = hashlib.sha256(f"{int(size)}:{suffix.lower()}".encode("utf-8")).hexdigest()
    return f"{SAFE_HASH_PREFIX}{digest[:10]}"


def test_safe_input_hash_is_deterministic_and_non_identifying():
    assert safe_input_hash(1234, ".txt") == _expected_safe_hash(1234, ".txt")
    assert safe_input_hash(1234, ".TXT") == _expected_safe_hash(1234, ".txt")
    assert safe_input_hash(1234, ".pdf") != safe_input_hash(1234, ".txt")
    assert safe_input_hash(0, "") == _expected_safe_hash(0, "noext")


def test_build_report_synthetic_fallback_passes_when_all_exit_codes_zero():
    handle = safe_input_hash(200, ".txt")
    report = build_report(
        branch="clinical-knowledge-architecture",
        head="29cb7da",
        dependency_prepare_exit_code=0,
        selected_input_mode="synthetic_fallback",
        input_suffix=".txt",
        input_safe_hash=handle,
        one_doc_validation_exit_code=0,
        streamlit_smoke_exit_code=0,
    )
    f = report["audit_findings"]
    assert report["passed"] is True
    assert f["selected_input_mode"] == "synthetic_fallback"
    assert f["input_suffix"] == ".txt"
    assert f["input_safe_hash"].startswith(SAFE_HASH_PREFIX)
    assert f["next_ui_command"] == NEXT_UI_COMMAND
    assert report["external_api_used"] is False
    assert report["auto_accept_enabled"] is False
    assert report["privacy_check_passed"] is True


def test_build_report_selected_file_mode():
    handle = safe_input_hash(4096, ".pdf")
    report = build_report(
        branch="clinical-knowledge-architecture",
        head="29cb7da",
        dependency_prepare_exit_code=0,
        selected_input_mode="selected_file",
        input_suffix=".pdf",
        input_safe_hash=handle,
        one_doc_validation_exit_code=0,
        streamlit_smoke_exit_code=0,
    )
    f = report["audit_findings"]
    assert f["selected_input_mode"] == "selected_file"
    assert f["input_suffix"] == ".pdf"
    assert f["input_safe_hash"] == handle


def test_build_report_rejects_unknown_input_mode():
    with pytest.raises(ValueError):
        build_report(
            branch="x",
            head="y",
            dependency_prepare_exit_code=0,
            selected_input_mode="autoselected_from_downloads",
            input_suffix=".txt",
            input_safe_hash=safe_input_hash(0, ".txt"),
            one_doc_validation_exit_code=0,
            streamlit_smoke_exit_code=0,
        )


def test_build_report_rejects_unknown_input_suffix():
    with pytest.raises(ValueError):
        build_report(
            branch="x",
            head="y",
            dependency_prepare_exit_code=0,
            selected_input_mode="selected_file",
            input_suffix=".jpg",
            input_safe_hash=safe_input_hash(0, ".jpg"),
            one_doc_validation_exit_code=0,
            streamlit_smoke_exit_code=0,
        )


def test_build_report_rejects_unsafe_hash_input():
    # A raw filename or path must not be accepted as a safe hash.
    for bad in (
        "patient_record.pdf",
        "/Users/operator/Documents/file.pdf",
        "C:\\Users\\operator\\file.pdf",
        "../etc/passwd",
        "oneclick_../etc",
        "noprefix",
    ):
        with pytest.raises(ValueError):
            build_report(
                branch="x",
                head="y",
                dependency_prepare_exit_code=0,
                selected_input_mode="selected_file",
                input_suffix=".pdf",
                input_safe_hash=bad,
                one_doc_validation_exit_code=0,
                streamlit_smoke_exit_code=0,
            )


def test_build_report_propagates_exit_codes_to_pass_state():
    handle = safe_input_hash(200, ".txt")
    # dependency failed
    report = build_report(
        branch="x",
        head="y",
        dependency_prepare_exit_code=1,
        selected_input_mode="synthetic_fallback",
        input_suffix=".txt",
        input_safe_hash=handle,
        one_doc_validation_exit_code=0,
        streamlit_smoke_exit_code=0,
    )
    assert report["passed"] is False
    # one-doc failed
    report = build_report(
        branch="x",
        head="y",
        dependency_prepare_exit_code=0,
        selected_input_mode="synthetic_fallback",
        input_suffix=".txt",
        input_safe_hash=handle,
        one_doc_validation_exit_code=2,
        streamlit_smoke_exit_code=0,
    )
    assert report["passed"] is False
    # smoke failed
    report = build_report(
        branch="x",
        head="y",
        dependency_prepare_exit_code=0,
        selected_input_mode="synthetic_fallback",
        input_suffix=".txt",
        input_safe_hash=handle,
        one_doc_validation_exit_code=0,
        streamlit_smoke_exit_code=3,
    )
    assert report["passed"] is False


def test_build_report_uses_existing_05_one_doc_conclusion():
    one_doc_json = (
        REPO_ROOT
        / "reports"
        / "medai_local_one_doc_operator_validation_05"
        / "medai_local_one_doc_operator_validation_05_report.json"
    )
    assert one_doc_json.is_file()
    payload = json.loads(one_doc_json.read_text(encoding="utf-8"))
    expected_conclusion = payload["conclusion"]
    handle = safe_input_hash(200, ".txt")
    report = build_report(
        branch="clinical-knowledge-architecture",
        head="29cb7da",
        dependency_prepare_exit_code=0,
        selected_input_mode="synthetic_fallback",
        input_suffix=".txt",
        input_safe_hash=handle,
        one_doc_validation_exit_code=0,
        streamlit_smoke_exit_code=0,
    )
    assert report["audit_findings"]["one_doc_validation_conclusion"] == expected_conclusion


def test_write_reports_produces_three_files_and_privacy_passes():
    handle = safe_input_hash(200, ".txt")
    report = build_report(
        branch="clinical-knowledge-architecture",
        head="29cb7da",
        dependency_prepare_exit_code=0,
        selected_input_mode="synthetic_fallback",
        input_suffix=".txt",
        input_safe_hash=handle,
        one_doc_validation_exit_code=0,
        streamlit_smoke_exit_code=0,
    )
    write_reports(report)
    assert JSON_REPORT_PATH.is_file()
    assert MD_REPORT_PATH.is_file()
    assert SHORT_MD_PATH.is_file()

    from clinical_knowledge.privacy import check_public_report_payload

    for path in (JSON_REPORT_PATH, MD_REPORT_PATH, SHORT_MD_PATH):
        content = path.read_text(encoding="utf-8")
        target = json.loads(content) if path.suffix == ".json" else content
        result = check_public_report_payload(target)
        assert result.passed, f"{path.name}: {result.leak_examples_redacted}"


def test_report_md_contains_next_ui_command_only():
    write_reports(
        build_report(
            branch="clinical-knowledge-architecture",
            head="29cb7da",
            dependency_prepare_exit_code=0,
            selected_input_mode="synthetic_fallback",
            input_suffix=".txt",
            input_safe_hash=safe_input_hash(200, ".txt"),
            one_doc_validation_exit_code=0,
            streamlit_smoke_exit_code=0,
        )
    )
    body = MD_REPORT_PATH.read_text(encoding="utf-8")
    assert NEXT_UI_COMMAND in body
    # Must not contain any browser-control instructions or absolute paths.
    for forbidden in ("Downloads", "Desktop", "Documents", "C:\\", "/Users/", "/home/"):
        assert forbidden not in body, forbidden


def test_powershell_orchestrator_static_contract():
    assert PS_ORCHESTRATOR.is_file()
    body = PS_ORCHESTRATOR.read_text(encoding="utf-8")
    # Required behaviors.
    for required in (
        "OpenFileDialog",
        "*.pdf",
        "*.txt",
        "one_click_input.pdf",
        "one_click_input.txt",
        "one_click_synthetic_lab.txt",
        "local_runtime_prepare_minimum_05.ps1",
        "run_medai_local_one_doc_operator_validation_05.py",
        "run_medai_streamlit_launch_smoke_05.py",
        "run_medai_local_one_click_operator_validation_06_report.py",
        "clinical-knowledge-architecture",
        "Multiselect",
    ):
        assert required in body, required
    # Must not auto-scan operator folders.
    for forbidden in (
        "Get-ChildItem -Recurse -Path C:",
        "Get-ChildItem -Recurse -Path $env:USERPROFILE",
        "$env:USERPROFILE\\Downloads",
        "$env:USERPROFILE\\Desktop",
        "$env:USERPROFILE\\Documents",
        "Start-Process iexplore",
        "Start-Process chrome",
        "Start-Process msedge",
    ):
        assert forbidden not in body, forbidden
    # Must not stage or commit through this script.
    assert "git add" not in body
    assert "git commit" not in body
    assert "git push" not in body


def test_no_cloud_sdks_introduced_by_new_scripts():
    for path in (PS_ORCHESTRATOR, REPORT_HELPER):
        body = path.read_text(encoding="utf-8")
        for forbidden in (
            "google-generativeai",
            "anthropic",
            "openai-python",
            "import openai",
            "from openai",
            "import anthropic",
            "from anthropic",
            "google.generativeai",
        ):
            assert forbidden not in body, f"{path.name}: {forbidden}"


def test_input_safe_hash_never_contains_path_separator():
    for size in (0, 1, 200, 4096, 99999):
        for suffix in (".pdf", ".txt"):
            h = safe_input_hash(size, suffix)
            assert "/" not in h
            assert "\\" not in h
            assert h.startswith(SAFE_HASH_PREFIX)


def test_existing_05_chain_tests_still_pass():
    import subprocess

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_medai_local_one_doc_operator_validation_05.py",
            "tests/test_extracted_medical_facts.py",
            "tests/test_run_review_extracted_information_preview.py",
            "tests/test_operator_review_actions.py",
            "-q",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
