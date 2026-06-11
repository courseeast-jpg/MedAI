"""Focused tests for MEDAI-OPERATOR-FIRST-UI-RESTORE-11C."""
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

SYNTHETIC_TEXT = (
    "Lab result report\n"
    "Glucose: 5.4 mmol/L (ref 3.9-5.5) [normal]\n"
    "Hemoglobin: 13.5 g/dL (ref 13.5-17.5)\n"
    "WBC: 7.2 x10E9/L (ref 4.0-11.0)\n"
    "Platelets: 230 x10E9/L (ref 150-450)\n"
)


@pytest.fixture()
def sql_store(tmp_path: Path):
    from mkb.sqlite_store import SQLiteStore

    return SQLiteStore(db_path=tmp_path / "mkb.db", encryption_key="")


@pytest.fixture()
def fallback_result(sql_store):
    from app.local_adapter_fallback_processor import process_adapter_fallback_run_review

    return process_adapter_fallback_run_review(
        sql_store,
        raw_text=SYNTHETIC_TEXT,
        selected_specialty="dermatology",
        session_id="test-11c",
    )


def test_default_ui_tabs_are_operator_first():
    from app.operator_ui_model import build_operator_ui_model

    model = build_operator_ui_model(show_advanced_tools=False)
    assert list(model.visible_tabs) == ["Run & Review", "MKB Explorer", "Review Queue"]


def test_advanced_engineering_tabs_are_hidden_by_default():
    from app.operator_ui_model import advanced_only_tabs, build_operator_ui_model

    model = build_operator_ui_model(show_advanced_tools=False)
    assert all(tab not in model.visible_tabs for tab in advanced_only_tabs())
    assert "Operator Control Panel" in build_operator_ui_model(show_advanced_tools=True).visible_tabs
    assert "Terminology Admin" in build_operator_ui_model(show_advanced_tools=True).visible_tabs


def test_run_review_exposes_medical_specialty_domain_by_default():
    from app.operator_ui_model import build_operator_ui_model

    source = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")
    model = build_operator_ui_model(show_advanced_tools=False)
    assert "Medical specialty / domain" in model.run_review_required_controls
    assert "Medical specialty / domain" in source
    assert "Used only to organize extracted facts in the MKB. It does not diagnose or interpret results." in source
    run_review_block = source[source.index("def render_current_run_tab") : source.index("def render_run_review_tab")]
    assert run_review_block.index('"Document category"') < run_review_block.index("render_specialty_selector(")


def test_specialty_options_include_required_operator_choices():
    from app.specialty_selection import specialty_labels_for_ui

    labels = specialty_labels_for_ui()
    assert "Urology" in labels
    assert "Dermatology" in labels
    assert "Gastroenterology" in labels
    assert "Other / needs review" in labels


def test_mkb_explorer_is_primary():
    from app.operator_ui_model import build_operator_ui_model

    assert "MKB Explorer" in build_operator_ui_model(show_advanced_tools=False).visible_tabs


def test_review_queue_is_primary():
    from app.operator_ui_model import build_operator_ui_model

    assert "Review Queue" in build_operator_ui_model(show_advanced_tools=False).visible_tabs


def test_mkb_explorer_exposes_tier_status_specialty_fields(sql_store, fallback_result):
    from app.mkb_explorer_model import build_mkb_explorer_model

    model = build_mkb_explorer_model(sql_store, specialty_filter="dermatology")
    assert model["row_count"] >= 4
    assert all({"specialty", "specialty_label", "tier", "status", "requires_review"}.issubset(row) for row in model["rows"])


def test_review_queue_exposes_actions_with_disclaimer():
    from app.operator_ui_model import SOURCE_COMPARISON_DISCLAIMER, build_operator_ui_model

    source = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")
    model = build_operator_ui_model(show_advanced_tools=False)
    assert {"Accept", "Reject", "Defer", "Source comparison disclaimer"}.issubset(model.review_queue_required_controls)
    assert SOURCE_COMPARISON_DISCLAIMER in source
    assert "review_queue_accept_" in source
    assert "review_queue_reject_" in source
    assert "review_queue_defer_" in source


def test_adapter_fallback_still_persists_selected_specialty(sql_store, fallback_result):
    ids = fallback_result["run_item"]["extracted_medical_fact_record_ids"]
    assert ids
    for record_id in ids:
        record = sql_store.get_record(record_id)
        assert record is not None
        assert record.specialty == "dermatology"


def test_records_remain_review_bound(sql_store, fallback_result):
    ids = fallback_result["run_item"]["extracted_medical_fact_record_ids"]
    for record_id in ids:
        record = sql_store.get_record(record_id)
        assert record is not None
        assert record.tier == "quarantined"
        assert record.requires_review is True
        assert record.structured["auto_accept_allowed"] is False


def test_external_api_false(fallback_result):
    assert fallback_result["external_api_used"] is False


def test_auto_accept_false(fallback_result):
    assert fallback_result["auto_accept_enabled"] is False


def test_validation_script_reports_privacy_safe():
    env = os.environ.copy()
    env.update({"MEDAI_ALLOW_EXTERNAL_API": "false", "MEDAI_LOCAL_ONLY": "true"})
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_medai_operator_first_ui_restore_11c.py")],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report_path = REPO_ROOT / "reports" / "medai_operator_first_ui_restore_11c" / "medai_operator_first_ui_restore_11c_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["conclusion"] == "operator_first_ui_restore_11c_ready"
    assert payload["privacy_check_passed"] is True
    assert payload["external_api_used"] is False
    assert payload["auto_accept_enabled"] is False
    rendered = report_path.read_text(encoding="utf-8")
    assert "C:\\Users" not in rendered
    assert "G:\\Codex" not in rendered


def test_block_10_focused_test_still_passes():
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_medai_ui_adapter_fallback_run_review_10.py", "-q"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_block_11b_test_still_passes_if_fast():
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_medai_ui_capability_restore_11b.py", "-q"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
