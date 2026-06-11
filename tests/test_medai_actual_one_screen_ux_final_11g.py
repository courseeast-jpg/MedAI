"""Source-level tests for MEDAI-ACTUAL-ONE-SCREEN-UX-FINAL-11G."""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MAIN_PATH = REPO_ROOT / "app" / "main.py"
STYLE_PATH = REPO_ROOT / "app" / "operator_compact_styles.py"
SOURCE = MAIN_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def _function_source(name: str) -> str:
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(SOURCE, node) or ""
    raise AssertionError(f"{name} not found")


def test_build_audit_details_rendered_only_when_advanced_enabled():
    main_source = _function_source("main")
    safety_source = _function_source("render_operator_safety_panel")
    assert "show_build_details=show_advanced_tools" in main_source
    assert "if not show_build_details:" in safety_source
    assert 'st.expander("Build / audit details", expanded=False)' in safety_source


def test_advanced_tabs_not_rendered_in_default_tab_list():
    main_source = _function_source("main")
    tab_block = main_source[main_source.index("tab_labels = [") : main_source.index("tabs = st.tabs(tab_labels)")]
    default_block = tab_block[: tab_block.index("if show_advanced_tools:")]
    assert "RUN_REVIEW_TAB" in default_block
    assert "MKB_EXPLORER_TAB" in default_block
    assert "REVIEW_QUEUE_TAB" in default_block
    assert "Operator Control Panel" not in default_block
    assert "Validation Batch Audit" not in default_block


def test_default_tabs_are_operator_tabs():
    main_source = _function_source("main")
    assert "RUN_REVIEW_TAB" in main_source
    assert "MKB_EXPLORER_TAB" in main_source
    assert "REVIEW_QUEUE_TAB" in main_source


def test_compact_style_helper_is_applied():
    style_source = STYLE_PATH.read_text(encoding="utf-8")
    assert "COMPACT_OPERATOR_CSS" in SOURCE
    assert "st.markdown(COMPACT_OPERATOR_CSS" in SOURCE
    assert "padding-top: .35rem" in style_source
    assert "compact-session-header" in style_source
    assert 'div[data-testid="stFileUploader"] section' in style_source


def test_result_guide_and_advanced_actions_collapsed_by_default():
    assert 'st.expander("Result guide", expanded=False)' in SOURCE
    assert 'st.expander("Advanced actions", expanded=False)' in SOURCE


def test_run_review_contains_category_specialty_upload_start():
    run_source = _function_source("render_current_run_tab")
    review_source = _function_source("render_run_review_tab")
    assert '"Document category"' in run_source
    assert "render_specialty_selector(" in run_source
    assert "Choose files" in run_source
    assert "Start run" in run_source
    assert "category_col, specialty_col = st.columns(2)" in run_source
    assert "upload_col, start_col = st.columns([3, 1])" in run_source
    assert "compact-workflow-row" in review_source


def test_mkb_explorer_and_review_queue_remain_actual_tabs():
    main_source = _function_source("main")
    assert "render_mkb_tab(sys_components)" in main_source
    assert "render_review_queue_tab(sys_components)" in main_source


def test_specialty_options_still_include_required_choices():
    from app.specialty_selection import specialty_labels_for_ui

    labels = set(specialty_labels_for_ui())
    assert {"Urology", "Dermatology", "Gastroenterology"}.issubset(labels)


def test_no_non_ui_logic_files_are_part_of_11g_scope():
    allowed = {
        "app/main.py",
        "app/operator_compact_styles.py",
        "tests/test_medai_actual_one_screen_ux_final_11g.py",
        "scripts/run_medai_actual_one_screen_ux_final_11g.py",
    }
    protected_prefixes = ("execution/", "mkb/", "clinical_knowledge/")
    assert not any(path.startswith(protected_prefixes) for path in allowed)


def test_external_api_auto_accept_false_and_reports_privacy_safe():
    env = os.environ.copy()
    env.update({"MEDAI_ALLOW_EXTERNAL_API": "false", "MEDAI_LOCAL_ONLY": "true"})
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_medai_actual_one_screen_ux_final_11g.py")],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report_path = REPO_ROOT / "reports" / "medai_actual_one_screen_ux_final_11g" / "medai_actual_one_screen_ux_final_11g_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["conclusion"] == "actual_one_screen_ux_final_11g_ready"
    assert payload["privacy_check_passed"] is True
    assert payload["external_api_used"] is False
    assert payload["auto_accept_enabled"] is False
    rendered = report_path.read_text(encoding="utf-8")
    assert "C:\\Users" not in rendered
    assert "G:\\Codex" not in rendered
