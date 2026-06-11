"""Focused tests for MEDAI-OPERATOR-ONE-SCREEN-UI-POLISH-11E."""
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
        session_id="test-11e",
    )


def test_compact_ui_model_exists():
    from app.operator_compact_ui_model import build_compact_operator_ui_model

    model = build_compact_operator_ui_model()
    assert model.compact_header_items


def test_header_chips_include_required_safety_states():
    from app.operator_compact_ui_model import build_compact_operator_ui_model

    items = set(build_compact_operator_ui_model().compact_header_items)
    assert {"Local safe mode", "Human review", "Local only", "Cloud APIs off", "Privacy check on"}.issubset(items)


def test_run_review_first_viewport_includes_core_controls():
    from app.operator_compact_ui_model import build_compact_operator_ui_model

    controls = set(build_compact_operator_ui_model().run_review_first_viewport_controls)
    assert {
        "Document category",
        "Medical specialty / domain",
        "Upload files",
        "Start run",
        "Documents waiting",
        "Current run status",
    }.issubset(controls)


def test_mkb_explorer_first_viewport_includes_counts_and_filters():
    from app.operator_compact_ui_model import build_compact_operator_ui_model

    controls = set(build_compact_operator_ui_model().mkb_explorer_first_viewport_controls)
    assert {
        "Total",
        "Active",
        "Quarantined / review-bound",
        "Superseded / rejected",
        "Specialty/domain filter",
        "Tier/status filter",
        "Fact type filter",
    }.issubset(controls)


def test_review_queue_first_viewport_includes_accept_reject_defer():
    from app.operator_compact_ui_model import build_compact_operator_ui_model

    controls = set(build_compact_operator_ui_model().review_queue_first_viewport_controls)
    assert {"Needs review count", "Source comparison disclaimer", "Accept", "Reject", "Defer"}.issubset(controls)


def test_advanced_engineering_tabs_hidden_by_default():
    from app.operator_ui_model import advanced_only_tabs, build_operator_ui_model

    model = build_operator_ui_model(show_advanced_tools=False)
    assert all(tab not in model.visible_tabs for tab in advanced_only_tabs())


def test_operator_tabs_remain_first_when_advanced_enabled():
    from app.operator_ui_model import build_operator_ui_model

    model = build_operator_ui_model(show_advanced_tools=True)
    assert list(model.visible_tabs[:3]) == ["Run & Review", "MKB Explorer", "Review Queue"]


def test_build_audit_details_hidden_by_default():
    source = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert "show_build_details=show_advanced_tools" in source
    assert "compact-session-header" in source


def test_specialty_options_still_include_required_choices():
    from app.specialty_selection import specialty_labels_for_ui

    labels = specialty_labels_for_ui()
    assert "Urology" in labels
    assert "Dermatology" in labels
    assert "Gastroenterology" in labels


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


def test_public_reports_privacy_safe():
    env = os.environ.copy()
    env.update({"MEDAI_ALLOW_EXTERNAL_API": "false", "MEDAI_LOCAL_ONLY": "true"})
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_medai_operator_one_screen_ui_polish_11e.py")],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report_path = REPO_ROOT / "reports" / "medai_operator_one_screen_ui_polish_11e" / "medai_operator_one_screen_ui_polish_11e_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["conclusion"] == "operator_one_screen_ui_polish_11e_ready"
    assert payload["privacy_check_passed"] is True
    assert payload["external_api_used"] is False
    assert payload["auto_accept_enabled"] is False
    rendered = report_path.read_text(encoding="utf-8")
    assert "C:\\Users" not in rendered
    assert "G:\\Codex" not in rendered


def test_11c_focused_tests_still_pass():
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_medai_operator_first_ui_restore_11c.py", "-q"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
