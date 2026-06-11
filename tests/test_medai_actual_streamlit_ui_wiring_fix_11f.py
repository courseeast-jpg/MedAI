"""Source-level tests for MEDAI-ACTUAL-STREAMLIT-UI-WIRING-FIX-11F."""
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
SOURCE = MAIN_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)

SPECIALTY_NAMES = {"Urology", "Dermatology", "Gastroenterology", "Neurology"}
SYNTHETIC_TEXT = (
    "Lab result report\n"
    "Glucose: 5.4 mmol/L (ref 3.9-5.5) [normal]\n"
    "Hemoglobin: 13.5 g/dL (ref 13.5-17.5)\n"
    "WBC: 7.2 x10E9/L (ref 4.0-11.0)\n"
    "Platelets: 230 x10E9/L (ref 150-450)\n"
)


def _literal_assigned_list(name: str) -> list[str]:
    for node in TREE.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return list(ast.literal_eval(node.value))
    raise AssertionError(f"{name} not found")


def _function_source(name: str) -> str:
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(SOURCE, node) or ""
    raise AssertionError(f"{name} not found")


def test_actual_main_tab_construction_includes_mkb_explorer():
    main_source = _function_source("main")
    assert "MKB_EXPLORER_TAB" in main_source
    assert "render_mkb_tab(sys_components)" in main_source


def test_actual_main_tab_construction_includes_review_queue():
    main_source = _function_source("main")
    assert "REVIEW_QUEUE_TAB" in main_source
    assert "render_review_queue_tab(sys_components)" in main_source


def test_actual_tab_order_places_operator_tabs_before_advanced_tabs():
    main_source = _function_source("main")
    run_index = main_source.index("RUN_REVIEW_TAB")
    mkb_index = main_source.index("MKB_EXPLORER_TAB")
    queue_index = main_source.index("REVIEW_QUEUE_TAB")
    advanced_index = main_source.index('"Operator Control Panel"')
    assert run_index < mkb_index < queue_index < advanced_index


def test_actual_main_contains_required_control_labels():
    assert '"Medical specialty / domain"' in SOURCE
    assert '"Document category"' in SOURCE


def test_document_category_options_do_not_contain_specialties():
    document_categories = set(_literal_assigned_list("DOCUMENT_CATEGORY_OPTIONS"))
    assert not (document_categories & SPECIALTY_NAMES)
    assert {"General", "Lab result", "Urinalysis", "Imaging report", "Other / needs review"}.issubset(document_categories)


def test_specialty_domain_options_do_contain_required_specialties():
    from app.specialty_selection import specialty_labels_for_ui

    labels = set(specialty_labels_for_ui())
    assert SPECIALTY_NAMES.issubset(labels)


def test_adapter_fallback_wires_selected_specialty():
    fallback_source = _function_source("render_adapter_fallback_panel")
    assert "selected_specialty=selected_specialty" in fallback_source
    assert "render_specialty_selector(" in fallback_source


def test_mkb_explorer_model_works_with_execution_none_and_sql_present(tmp_path: Path):
    from app.local_adapter_fallback_processor import process_adapter_fallback_run_review
    from app.mkb_explorer_model import build_mkb_explorer_model
    from mkb.sqlite_store import SQLiteStore

    sql_store = SQLiteStore(db_path=tmp_path / "mkb.db", encryption_key="")
    process_adapter_fallback_run_review(sql_store, raw_text=SYNTHETIC_TEXT, selected_specialty="dermatology")
    sys_components = {"execution": None, "sql": sql_store}
    model = build_mkb_explorer_model(sys_components["sql"], specialty_filter="dermatology")
    assert sys_components["execution"] is None
    assert model["available"] is True
    assert model["row_count"] >= 4
    assert all({"specialty", "tier", "status", "requires_review"}.issubset(row) for row in model["rows"])


def test_review_queue_model_works_with_execution_none_and_sql_present(tmp_path: Path):
    from app.local_adapter_fallback_processor import process_adapter_fallback_run_review
    from app.mkb_explorer_model import build_mkb_explorer_model
    from mkb.sqlite_store import SQLiteStore

    sql_store = SQLiteStore(db_path=tmp_path / "mkb.db", encryption_key="")
    process_adapter_fallback_run_review(sql_store, raw_text=SYNTHETIC_TEXT, selected_specialty="dermatology")
    sys_components = {"execution": None, "sql": sql_store}
    model = build_mkb_explorer_model(sys_components["sql"], tier_filter="review_bound")
    assert sys_components["execution"] is None
    assert model["available"] is True
    assert model["row_count"] >= 4
    assert all(row["requires_review"] for row in model["rows"])


def test_external_api_and_auto_accept_false(tmp_path: Path):
    from app.local_adapter_fallback_processor import process_adapter_fallback_run_review
    from mkb.sqlite_store import SQLiteStore

    sql_store = SQLiteStore(db_path=tmp_path / "mkb.db", encryption_key="")
    result = process_adapter_fallback_run_review(sql_store, raw_text=SYNTHETIC_TEXT, selected_specialty="dermatology")
    assert result["external_api_used"] is False
    assert result["auto_accept_enabled"] is False


def test_reports_privacy_safe():
    env = os.environ.copy()
    env.update({"MEDAI_ALLOW_EXTERNAL_API": "false", "MEDAI_LOCAL_ONLY": "true"})
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_medai_actual_streamlit_ui_wiring_fix_11f.py")],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report_path = REPO_ROOT / "reports" / "medai_actual_streamlit_ui_wiring_fix_11f" / "medai_actual_streamlit_ui_wiring_fix_11f_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["conclusion"] == "actual_streamlit_ui_wiring_fix_11f_ready"
    assert payload["privacy_check_passed"] is True
    assert payload["external_api_used"] is False
    assert payload["auto_accept_enabled"] is False
    rendered = report_path.read_text(encoding="utf-8")
    assert "C:\\Users" not in rendered
    assert "G:\\Codex" not in rendered


def test_11e_tests_still_pass():
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_medai_operator_one_screen_ui_polish_11e.py", "-q"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=150,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
