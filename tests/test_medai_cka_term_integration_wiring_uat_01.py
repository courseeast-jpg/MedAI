"""Tests for MEDAI-CKA-TERM-INTEGRATION-WIRING-UAT-01."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clinical_knowledge.privacy.report_privacy import check_public_report_payload  # noqa: E402
from scripts.run_medai_cka_term_integration_wiring_uat_01 import (  # noqa: E402
    load_plan_function,
    run_uat,
    safe_item,
    unsafe_item,
    TERMINOLOGY_LOOKUP_ENV_VAR,
    UI_ENV_VAR,
)

SCOPED_FILES = {
    "scripts/run_medai_cka_term_integration_wiring_uat_01.py",
    "tests/test_medai_cka_term_integration_wiring_uat_01.py",
    "reports/medai_cka_term_integration_wiring_uat_01/MEDAI_CKA_TERM_INTEGRATION_WIRING_UAT_01.md",
    "reports/medai_cka_term_integration_wiring_uat_01/medai_cka_term_integration_wiring_uat_01_report.json",
    "reports/medai_cka_term_integration_wiring_uat_01/medai_cka_term_integration_wiring_uat_01_report.md",
}


def test_uat_modifies_no_runtime_files() -> None:
    result = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    changed = {line[3:].replace("\\", "/") for line in result.stdout.splitlines() if line.strip()}
    assert changed <= SCOPED_FILES


def test_app_main_not_modified_by_this_uat_block() -> None:
    result = subprocess.run(
        ["git", "diff", "--", "app/main.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout == ""


def test_cka_terminology_wiring_block_exists_once() -> None:
    source = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert source.count("MEDAI-CKA-TERM-INTEGRATION-WIRING-NEXT-01") == 2
    assert source.count("terminology_match_hypothesis_ui_plan(item)") == 1


def test_env_gate_render_call_counts() -> None:
    report = run_uat()
    assert report["neither_env_truthy_render_calls"] == 0
    assert report["helper_only_env_truthy_render_calls"] == 0
    assert report["ui_only_env_truthy_render_calls"] == 0
    assert report["both_env_truthy_allowed_calls_only"] is True


def test_both_envs_truthy_safe_metadata_render_plan() -> None:
    plan = load_plan_function()(
        safe_item(),
        env={TERMINOLOGY_LOOKUP_ENV_VAR: "true", UI_ENV_VAR: "true"},
    )
    assert plan is not None
    assert set(plan) == {"expander_label", "markdown_lines", "disclaimer_line", "env_vars"}
    rendered = "\n".join(plan["markdown_lines"])
    assert "exact_terminology_match" in rendered
    assert "Auto-accept allowed" in rendered


def test_unsafe_metadata_does_not_render() -> None:
    plan = load_plan_function()(
        unsafe_item(),
        env={TERMINOLOGY_LOOKUP_ENV_VAR: "true", UI_ENV_VAR: "true"},
    )
    assert plan is None
    assert run_uat()["unsafe_metadata_render_count"] == 0


def test_no_licensed_row_or_raw_private_content_renders() -> None:
    report = run_uat()
    assert report["licensed_row_content_render_count"] == 0
    assert report["raw_text_render_count"] == 0
    assert report["raw_ocr_text_rendered"] is False
    assert report["raw_document_text_rendered"] is False
    assert report["raw_filenames_rendered"] is False
    assert report["private_path_render_count"] == 0
    assert report["private_paths_rendered"] is False


def test_no_forbidden_streamlit_or_mutating_calls() -> None:
    report = run_uat()
    assert report["allowed_streamlit_calls"] == ["st.markdown", "st.caption"]
    assert report["forbidden_streamlit_call_count"] == 0
    for key in (
        "streamlit_wiring_changed",
        "terminology_import_performed",
        "terminology_data_staged",
    ):
        assert report[key] is False


def test_nonclinical_review_bound_flags() -> None:
    report = run_uat()
    assert report["clinical_interpretation_performed"] is False
    assert report["diagnosis_inference_performed"] is False
    assert report["treatment_inference_performed"] is False
    assert report["medication_inference_performed"] is False
    assert report["ddi_behavior_changed"] is False
    assert report["external_api_used"] is False
    assert report["auto_accept_allowed"] is False
    assert report["review_bound_outputs"] is True
    assert report["review_required_count"] == 1


def test_freeze_and_park_tags_not_touched() -> None:
    for commit in ("7ef8ffd", "3e46461", "9f9e22d", "f4d3cc6", "748c32a"):
        result = subprocess.run(
            ["git", "tag", "--points-at", commit],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip(), f"missing expected tag at {commit}"


def test_cue_expansion_remains_not_recommended() -> None:
    report = run_uat()
    assert report["cue_expansion_recommended"] is False
    assert report["cue_expansion_performed"] is False


def test_report_payload_privacy_clean() -> None:
    assert check_public_report_payload(run_uat()).passed is True


def test_generated_reports_privacy_clean_if_present() -> None:
    report_dir = REPO_ROOT / "reports" / "medai_cka_term_integration_wiring_uat_01"
    if not report_dir.exists():
        pytest.skip("UAT reports not generated yet")
    for path in report_dir.iterdir():
        payload = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else {"report_text": path.read_text(encoding="utf-8")}
        assert check_public_report_payload(payload).passed is True
