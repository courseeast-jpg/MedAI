"""Focused tests for MEDAI-OPERATOR-USABILITY-POLISH-13C."""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

from clinical_knowledge.privacy import check_public_report_payload

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = REPO_ROOT / "app" / "main.py"
CSS_PATH = REPO_ROOT / "app" / "operator_compact_styles.py"
SOURCE = MAIN_PATH.read_text(encoding="utf-8")
CSS_SOURCE = CSS_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def _function_source(name: str) -> str:
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(SOURCE, node) or ""
    raise AssertionError(f"{name} not found")


def test_start_run_primary_button_style_marker_exists() -> None:
    run_source = _function_source("render_current_run_tab")

    assert "operator-start-rail" in run_source
    assert '"Start run"' in run_source
    assert 'type="primary"' in run_source
    assert "button[kind=\"primary\"]" in CSS_SOURCE
    assert "min-height: 3rem" in CSS_SOURCE


def test_start_run_disabled_reason_remains_visible_when_queue_empty() -> None:
    from app.main import queue_display_state, start_run_state_reason

    state = start_run_state_reason(queue_display_state(queued_count=0, selected_count=0))

    assert state["enabled"] is False
    assert state["reason"] == "No documents queued. Add supported files to start."


def test_start_run_enabled_reason_remains_visible_when_queue_has_files() -> None:
    from app.main import queue_display_state, start_run_state_reason

    state = start_run_state_reason(queue_display_state(queued_count=12, selected_count=0))

    assert state["enabled"] is True
    assert state["reason"] == "Ready: 12 documents waiting."


def test_add_selected_files_to_queue_is_prominent_when_files_selected() -> None:
    run_source = _function_source("render_current_run_tab")

    assert "operator-add-queue-callout" in run_source
    assert "Next step: add selected files to the queue." in run_source
    assert '"Add selected files to queue", type="primary", use_container_width=True' in run_source


def test_advanced_actions_remains_collapsed_by_default() -> None:
    run_source = _function_source("render_current_run_tab")

    assert 'st.expander("Advanced actions", expanded=False)' in run_source


def test_result_guide_remains_collapsed_by_default() -> None:
    guidance_source = _function_source("render_operator_guidance_panel")

    assert 'st.expander("Result guide", expanded=False)' in guidance_source


def test_previous_review_summary_remains_collapsed_by_default() -> None:
    run_review_source = _function_source("render_run_review_tab")

    assert 'st.expander("Previous review summary / aggregate review status", expanded=False)' in run_review_source


def test_safety_banner_remains_visible() -> None:
    safety_source = _function_source("render_operator_safety_panel")

    assert "Review required - not for diagnosis." in safety_source
    assert "MedAI does not diagnose" in SOURCE


def test_safety_pills_remain_visible_but_compact() -> None:
    safety_source = _function_source("render_operator_safety_panel")

    for expected in ("Local only", "Cloud APIs off", "Privacy check on", "Human review"):
        assert expected in safety_source
    assert "font-size: .72rem" in CSS_SOURCE


def test_no_auto_accept_copy_or_behavior_introduced() -> None:
    from app.main import operator_console_redesign_static_model

    assert operator_console_redesign_static_model()["auto_accept"] is False
    assert "No auto-accept" in SOURCE
    assert "automatic acceptance" in SOURCE


def test_mkb_explorer_still_shows_active_and_quarantined_review_bound() -> None:
    mkb_source = _function_source("render_mkb_tab")

    assert '"Active"' in mkb_source
    assert '"Quarantined / review-bound"' in mkb_source
    assert 'index=3 if base_counts["review_bound"] else 0' in mkb_source


def test_review_queue_still_shows_required_actions() -> None:
    review_source = _function_source("render_review_queue_tab")

    assert "Accept after source comparison" in SOURCE
    assert "reject_cfg" in review_source
    assert "defer_cfg" in review_source
    assert "Needs human review." in review_source


def test_functional_12a_uat_still_passes() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_medai_operator_workflow_uat_12a.py"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "medai_operator_workflow_uat_12a_ready" in proc.stdout


def test_privacy_report_safe() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_medai_operator_usability_polish_13c.py"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    report_dir = REPO_ROOT / "reports" / "medai_operator_usability_polish_13c"
    for path in report_dir.iterdir():
        payload = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else path.read_text(encoding="utf-8")
        result = check_public_report_payload(payload)
        assert result.passed, f"{path.name}: {result.leak_examples_redacted}"
