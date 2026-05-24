"""Focused tests for the 07 self-healing validation block."""
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

PS_RUNNER = REPO_ROOT / "scripts" / "run_medai_local_self_healing_validation_07.ps1"
PS_06_ORCHESTRATOR = REPO_ROOT / "scripts" / "run_medai_local_one_click_operator_validation_06.ps1"
AGGREGATOR = REPO_ROOT / "scripts" / "run_medai_local_self_healing_validation_07.py"
REPORT_DIR = REPO_ROOT / "reports" / "medai_local_self_healing_validation_07"
JSON_REPORT_PATH = REPORT_DIR / "medai_local_self_healing_validation_07_report.json"
MD_REPORT_PATH = REPORT_DIR / "medai_local_self_healing_validation_07_report.md"
SHORT_MD_PATH = REPORT_DIR / "MEDAI_LOCAL_SELF_HEALING_VALIDATION_07.md"
ONE_DOC_REPORT_JSON = (
    REPO_ROOT
    / "reports"
    / "medai_local_one_doc_operator_validation_05"
    / "medai_local_one_doc_operator_validation_05_report.json"
)

from scripts.run_medai_local_self_healing_validation_07 import (  # noqa: E402
    ALLOWED_INPUT_MODES,
    ALLOWED_INPUT_SUFFIXES,
    NEXT_UI_COMMAND,
    SAFE_HASH_PREFIX,
    build_report,
    run_adapter_diagnostic,
    safe_input_hash,
    write_reports,
)


def _ensure_one_doc_report_state(target_conclusion: str) -> None:
    """Mutate the 05 report file in place to simulate the requested
    upstream conclusion. Restored to the prior content by the test
    teardown via the ``one_doc_report_snapshot`` fixture below.
    """
    payload = json.loads(ONE_DOC_REPORT_JSON.read_text(encoding="utf-8"))
    payload["conclusion"] = target_conclusion
    findings = payload.setdefault("audit_findings", {})
    if target_conclusion == "one_doc_operator_validation_ready":
        findings["structured_facts_extracted_count"] = 3
        findings["review_bound_records_persisted_count"] = 3
        findings["retrieval_proof_count"] = 3
        findings["ui_render_plan_row_count"] = 3
    else:
        findings["structured_facts_extracted_count"] = 0
        findings["review_bound_records_persisted_count"] = 0
        findings["retrieval_proof_count"] = 0
        findings["ui_render_plan_row_count"] = 0
    ONE_DOC_REPORT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@pytest.fixture()
def one_doc_report_snapshot():
    """Snapshot the 05 report file and restore it after the test."""
    original = ONE_DOC_REPORT_JSON.read_text(encoding="utf-8")
    try:
        yield
    finally:
        ONE_DOC_REPORT_JSON.write_text(original, encoding="utf-8")


# ---------------------------------------------------------------------------
# Aggregator unit tests
# ---------------------------------------------------------------------------

def test_safe_input_hash_is_deterministic_and_prefixed():
    a = safe_input_hash(200, ".txt")
    assert a.startswith(SAFE_HASH_PREFIX)
    assert safe_input_hash(200, ".TXT") == a
    assert safe_input_hash(200, ".pdf") != a
    assert "/" not in a and "\\" not in a


def test_build_report_rejects_unknown_inputs():
    h = safe_input_hash(1, ".txt")
    for bad_mode in ("downloads", "auto_pick", "from_desktop", ""):
        with pytest.raises(ValueError):
            build_report(
                branch="x",
                head="y",
                dependency_prepare_result="ok",
                input_mode=bad_mode,
                input_suffix=".txt",
                input_safe_hash=h,
                streamlit_launch_attempted=False,
            )
    for bad_suffix in (".jpg", ".docx", ".tiff"):
        with pytest.raises(ValueError):
            build_report(
                branch="x",
                head="y",
                dependency_prepare_result="ok",
                input_mode="synthetic_fallback",
                input_suffix=bad_suffix,
                input_safe_hash=h,
                streamlit_launch_attempted=False,
            )
    for bad_hash in (
        "patient_record.pdf",
        "C:\\Users\\op\\private\\file.pdf",
        "/home/operator/private/file.pdf",
        "noprefix_abcdef",
        "selfheal_../etc",
    ):
        with pytest.raises(ValueError):
            build_report(
                branch="x",
                head="y",
                dependency_prepare_result="ok",
                input_mode="synthetic_fallback",
                input_suffix=".txt",
                input_safe_hash=bad_hash,
                streamlit_launch_attempted=False,
            )


def test_pipeline_path_ready_when_05_is_ready(one_doc_report_snapshot):
    _ensure_one_doc_report_state("one_doc_operator_validation_ready")
    h = safe_input_hash(200, ".txt")
    report = build_report(
        branch="clinical-knowledge-architecture",
        head="dev",
        dependency_prepare_result="ok",
        input_mode="selected_file",
        input_suffix=".txt",
        input_safe_hash=h,
        streamlit_launch_attempted=False,
    )
    f = report["audit_findings"]
    assert f["pipeline_path_ready"] is True
    assert f["adapter_path_ready"] is False
    assert f["final_conclusion"] == "local_operator_validation_ready_pipeline_path"
    assert f["next_ui_command"] == NEXT_UI_COMMAND
    assert report["external_api_used"] is False
    assert report["auto_accept_enabled"] is False
    assert report["privacy_check_passed"] is True


def test_adapter_path_fallback_when_05_is_not_ready(one_doc_report_snapshot):
    _ensure_one_doc_report_state("not_ready")
    h = safe_input_hash(200, ".txt")
    report = build_report(
        branch="clinical-knowledge-architecture",
        head="dev",
        dependency_prepare_result="ok",
        input_mode="synthetic_fallback",
        input_suffix=".txt",
        input_safe_hash=h,
        streamlit_launch_attempted=False,
    )
    f = report["audit_findings"]
    assert f["pipeline_path_ready"] is False
    assert f["adapter_path_ready"] is True
    assert f["final_conclusion"] == "local_operator_validation_ready_adapter_path"
    assert f["structured_facts_extracted"] >= 6
    assert f["review_bound_records_persisted"] >= 6
    assert f["retrieval_proof_count"] >= 6
    assert f["ui_preview_rows"] >= 6
    assert f["operator_action_proof_count"] == 3
    assert report["external_api_used"] is False
    assert report["auto_accept_enabled"] is False
    assert report["privacy_check_passed"] is True
    assert f.get("reason") == "pipeline_produced_zero_facts_adapter_path_validated"


def test_adapter_path_fallback_when_05_is_no_input_file(one_doc_report_snapshot):
    _ensure_one_doc_report_state("no_input_file")
    h = safe_input_hash(0, ".txt")
    report = build_report(
        branch="clinical-knowledge-architecture",
        head="dev",
        dependency_prepare_result="ok",
        input_mode="synthetic_fallback",
        input_suffix=".txt",
        input_safe_hash=h,
        streamlit_launch_attempted=False,
    )
    f = report["audit_findings"]
    assert f["pipeline_path_ready"] is False
    assert f["adapter_path_ready"] is True
    assert f["final_conclusion"] == "local_operator_validation_ready_adapter_path"


def test_force_adapter_diagnostic_overrides_ready_05(one_doc_report_snapshot):
    _ensure_one_doc_report_state("one_doc_operator_validation_ready")
    h = safe_input_hash(200, ".txt")
    report = build_report(
        branch="clinical-knowledge-architecture",
        head="dev",
        dependency_prepare_result="ok",
        input_mode="synthetic_fallback",
        input_suffix=".txt",
        input_safe_hash=h,
        streamlit_launch_attempted=False,
        force_adapter_diagnostic=True,
    )
    f = report["audit_findings"]
    assert f["pipeline_path_ready"] is False
    assert f["adapter_path_ready"] is True
    assert f["final_conclusion"] == "local_operator_validation_ready_adapter_path"


def test_run_adapter_diagnostic_extracts_english_and_russian_lab_facts():
    findings = run_adapter_diagnostic()
    assert findings["passed"] is True
    # English (3) + Russian (3) = at least 6
    assert findings["structured_facts_extracted"] >= 6
    assert findings["review_bound_records_persisted"] >= 6
    assert findings["retrieval_proof_count"] >= 6
    assert findings["ui_preview_rows"] >= 6
    assert findings["operator_action_proof_count"] == 3
    assert findings["ui_render_plan_built"] is True
    assert findings["external_api_used"] is False
    assert findings["auto_accept_enabled"] is False


def test_next_ui_command_is_exactly_safe():
    assert NEXT_UI_COMMAND == "streamlit run app/main.py"


def test_write_reports_produces_three_privacy_safe_files(one_doc_report_snapshot):
    _ensure_one_doc_report_state("not_ready")
    h = safe_input_hash(200, ".txt")
    report = build_report(
        branch="clinical-knowledge-architecture",
        head="dev",
        dependency_prepare_result="ok",
        input_mode="synthetic_fallback",
        input_suffix=".txt",
        input_safe_hash=h,
        streamlit_launch_attempted=False,
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


def test_report_md_contains_only_next_ui_command_for_browser(one_doc_report_snapshot):
    _ensure_one_doc_report_state("not_ready")
    h = safe_input_hash(200, ".txt")
    write_reports(
        build_report(
            branch="clinical-knowledge-architecture",
            head="dev",
            dependency_prepare_result="ok",
            input_mode="synthetic_fallback",
            input_suffix=".txt",
            input_safe_hash=h,
            streamlit_launch_attempted=False,
        )
    )
    body = MD_REPORT_PATH.read_text(encoding="utf-8")
    assert NEXT_UI_COMMAND in body
    for forbidden in (
        "Downloads",
        "Desktop",
        "Documents",
        "C:\\Users",
        "G:\\Codex",
        "/Users/",
        "/home/",
    ):
        assert forbidden not in body, forbidden


# ---------------------------------------------------------------------------
# Static PowerShell contract tests
# ---------------------------------------------------------------------------

def test_06_orchestrator_has_null_safe_git_status_handling():
    body = PS_06_ORCHESTRATOR.read_text(encoding="utf-8")
    # The null-safe pattern must appear: wrap git output with @() and join.
    assert "@(& git status --porcelain)" in body
    # And the array length guard must appear.
    assert "$line.Length -lt 4" in body


def test_07_ps_runner_static_contract():
    assert PS_RUNNER.is_file()
    body = PS_RUNNER.read_text(encoding="utf-8")
    # Required: file picker only, no auto-folder scans.
    for required in (
        "OpenFileDialog",
        "*.pdf",
        "*.txt",
        "self_healing_input.pdf",
        "self_healing_input.txt",
        "self_healing_synthetic_lab.txt",
        "local_runtime_prepare_minimum_05.ps1",
        "run_medai_local_one_doc_operator_validation_05.py",
        "run_medai_streamlit_launch_smoke_05.py",
        "run_medai_local_self_healing_validation_07.py",
        "clinical-knowledge-architecture",
        "Multiselect",
        "streamlit run app/main.py",
    ):
        assert required in body, required
    # No folder scans, no auto-browser launches.
    for forbidden in (
        "Get-ChildItem -Recurse -Path C:",
        "$env:USERPROFILE\\Downloads",
        "$env:USERPROFILE\\Desktop",
        "$env:USERPROFILE\\Documents",
        "Start-Process iexplore",
        "Start-Process chrome",
        "Start-Process msedge",
    ):
        assert forbidden not in body, forbidden
    # No git mutations from this script.
    assert "git add" not in body
    assert "git commit" not in body
    assert "git push" not in body


def test_07_no_cloud_sdks_introduced():
    for path in (PS_RUNNER, AGGREGATOR):
        body = path.read_text(encoding="utf-8")
        for forbidden in (
            "google-generativeai",
            "anthropic",
            "import openai",
            "from openai",
            "import anthropic",
            "from anthropic",
            "google.generativeai",
        ):
            assert forbidden not in body, f"{path.name}: {forbidden}"


# ---------------------------------------------------------------------------
# Cross-chain regression
# ---------------------------------------------------------------------------

def test_existing_06_and_chain_tests_still_pass():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_medai_local_one_click_operator_validation_06.py",
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
