"""Focused tests for MEDAI-UI-CAPABILITY-RESTORE-11B."""
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

REQUIRED_SPECIALTY_KEYS = {
    "general",
    "dermatology",
    "gastroenterology",
    "cardiology",
    "neurology",
    "endocrinology",
    "hematology",
    "nephrology",
    "pulmonology",
    "rheumatology",
    "infectious_disease",
    "oncology",
    "pediatrics",
    "obstetrics_gynecology",
    "psychiatry",
    "urology",
    "ophthalmology",
    "otolaryngology",
    "orthopedics",
    "other_review",
}

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
        session_id="test-11b",
    )


def test_specialty_option_list_contains_all_required_stable_keys():
    from app.specialty_selection import specialty_options_for_ui

    assert {item["key"] for item in specialty_options_for_ui()} == REQUIRED_SPECIALTY_KEYS


def test_default_specialty_is_general():
    from app.specialty_selection import DEFAULT_SPECIALTY_KEY, specialty_label

    assert DEFAULT_SPECIALTY_KEY == "general"
    assert specialty_label(None) == "General medicine"


def test_invalid_specialty_falls_back_to_general():
    from app.specialty_selection import validate_specialty_key

    assert validate_specialty_key("not-a-specialty") == "general"


def test_run_review_source_exposes_specialty_selector_visible_by_default():
    source = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert "Medical specialty / domain" in source
    assert "Used only to organize extracted facts in the MKB. It does not diagnose or interpret results." in source
    assert '"MKB Explorer"' in source
    assert "PRIMARY_OPERATOR_TABS" in source


def test_adapter_fallback_persists_selected_specialty(sql_store, fallback_result):
    ids = fallback_result["run_item"]["extracted_medical_fact_record_ids"]
    assert ids
    for record_id in ids:
        record = sql_store.get_record(record_id)
        assert record is not None
        assert record.specialty == "dermatology"


def test_persisted_records_remain_review_bound(sql_store, fallback_result):
    ids = fallback_result["run_item"]["extracted_medical_fact_record_ids"]
    for record_id in ids:
        record = sql_store.get_record(record_id)
        assert record is not None
        assert record.fact_type == "test_result"
        assert record.tier == "quarantined"
        assert record.requires_review is True
        assert record.structured["auto_accept_allowed"] is False


def test_mkb_explorer_model_shows_total_active_quarantined_counts(sql_store, fallback_result):
    from app.mkb_explorer_model import build_mkb_explorer_model

    model = build_mkb_explorer_model(sql_store)
    assert model["available"] is True
    assert model["counts"]["total"] >= 4
    assert model["counts"]["active"] == 0
    assert model["counts"]["quarantined"] >= 4
    assert model["counts"]["review_bound"] >= 4


def test_mkb_explorer_model_supports_specialty_filter(sql_store, fallback_result):
    from app.mkb_explorer_model import build_mkb_explorer_model

    model = build_mkb_explorer_model(sql_store, specialty_filter="dermatology")
    assert model["row_count"] >= 4
    assert all(row["specialty"] == "dermatology" for row in model["rows"])


def test_mkb_explorer_model_supports_tier_status_visibility(sql_store, fallback_result):
    from app.mkb_explorer_model import build_mkb_explorer_model

    model = build_mkb_explorer_model(sql_store, tier_filter="quarantined")
    assert model["row_count"] >= 4
    assert model["tier_status_visible"] is True
    assert all({"tier", "status", "requires_review"}.issubset(row) for row in model["rows"])


def test_mkb_explorer_does_not_require_execution_pipeline(sql_store, fallback_result):
    from app.mkb_explorer_model import build_mkb_explorer_model

    sys_components = {"sql": sql_store, "execution": None}
    model = build_mkb_explorer_model(sys_components["sql"], specialty_filter="dermatology")
    assert sys_components["execution"] is None
    assert model["available"] is True
    assert model["row_count"] >= 4


def test_accept_reject_defer_remain_available_for_review_bound_records(sql_store, fallback_result):
    from app.local_adapter_fallback_processor import run_operator_action_proof

    ids = fallback_result["run_item"]["extracted_medical_fact_record_ids"]
    assert run_operator_action_proof(sql_store, ids, session_id="test-11b") == 3


def test_no_raw_source_text_private_paths_or_phi_in_public_result(fallback_result):
    from clinical_knowledge.privacy import check_public_report_payload

    public = {
        "run_item": fallback_result["run_item"],
        "selected_specialty": fallback_result["selected_specialty"],
        "external_api_used": fallback_result["external_api_used"],
        "auto_accept_enabled": fallback_result["auto_accept_enabled"],
    }
    result = check_public_report_payload(public)
    assert result.passed, result.leak_examples_redacted


def test_external_api_false(fallback_result):
    assert fallback_result["external_api_used"] is False


def test_auto_accept_false(fallback_result):
    assert fallback_result["auto_accept_enabled"] is False


def test_block_10_focused_test_still_passes():
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_medai_ui_adapter_fallback_run_review_10.py", "-q"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_validation_script_reports_ready():
    env = os.environ.copy()
    env.update({"MEDAI_ALLOW_EXTERNAL_API": "false", "MEDAI_LOCAL_ONLY": "true"})
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_medai_ui_capability_restore_11b.py")],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(
        (REPO_ROOT / "reports" / "medai_ui_capability_restore_11b" / "medai_ui_capability_restore_11b_report.json").read_text(encoding="utf-8")
    )
    assert payload["conclusion"] == "ui_capability_restore_11b_ready"
    assert payload["privacy_check_passed"] is True
    assert payload["external_api_used"] is False
    assert payload["auto_accept_enabled"] is False
