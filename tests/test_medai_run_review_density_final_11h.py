"""Source-level tests for MEDAI-RUN-REVIEW-DENSITY-FINAL-11H."""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

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


def test_result_guide_previous_summary_and_advanced_actions_collapsed():
    assert 'st.expander("Result guide", expanded=False)' in SOURCE
    assert 'st.expander("Previous review summary / aggregate review status", expanded=False)' in SOURCE
    assert 'st.expander("Advanced actions", expanded=False)' in SOURCE


def test_no_run_state_does_not_render_large_zero_result_cards():
    run_source = _function_source("render_current_run_tab")
    assert "if active_run:" in run_source
    assert "render_run_status_panel(active_run" in run_source
    assert 'st.caption("No current run results' not in run_source


def test_queued_documents_support_compact_one_line_summary():
    queue_source = _function_source("render_queue_panel")
    assert "if len(files) == 1:" in queue_source
    assert 'row[0].caption(f"Documents waiting: {path.name}")' in queue_source
    assert 'st.caption(f"Documents waiting: {len(files)}")' in queue_source


def test_upload_start_category_and_specialty_remain_first_viewport():
    run_source = _function_source("render_current_run_tab")
    assert "category_col, specialty_col = st.columns(2)" in run_source
    assert "upload_col, start_col = st.columns([3, 1])" in run_source
    assert '"Document category"' in run_source
    assert "render_specialty_selector(" in run_source
    assert "Choose files" in run_source
    assert "Start run" in run_source


def test_mkb_explorer_and_review_queue_tabs_remain_visible():
    main_source = _function_source("main")
    assert "MKB_EXPLORER_TAB" in main_source
    assert "REVIEW_QUEUE_TAB" in main_source
    assert "render_mkb_tab(sys_components)" in main_source
    assert "render_review_queue_tab(sys_components)" in main_source


def test_specialty_options_still_include_required_choices():
    from app.specialty_selection import specialty_labels_for_ui

    labels = set(specialty_labels_for_ui())
    assert {"Urology", "Dermatology", "Gastroenterology"}.issubset(labels)


def test_compact_css_reduces_density_without_hiding_controls():
    style_source = STYLE_PATH.read_text(encoding="utf-8")
    assert "gap: .22rem" in style_source
    assert "min-height: 2.25rem" in style_source
    assert "button[kind=" in style_source
    assert "display: none" in style_source  # only uploader helper text is hidden by existing compact style


def test_external_api_auto_accept_false_and_reports_privacy_safe():
    env = os.environ.copy()
    env.update({"MEDAI_ALLOW_EXTERNAL_API": "false", "MEDAI_LOCAL_ONLY": "true"})
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_medai_run_review_density_final_11h.py")],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report_path = REPO_ROOT / "reports" / "medai_run_review_density_final_11h" / "medai_run_review_density_final_11h_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["conclusion"] == "run_review_density_final_11h_ready"
    assert payload["privacy_check_passed"] is True
    assert payload["external_api_used"] is False
    assert payload["auto_accept_enabled"] is False
    rendered = report_path.read_text(encoding="utf-8")
    assert "C:\\Users" not in rendered
    assert "G:\\Codex" not in rendered
