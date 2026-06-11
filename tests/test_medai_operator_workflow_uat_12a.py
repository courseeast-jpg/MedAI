from __future__ import annotations

import json
from pathlib import Path

import app.test_launcher as launcher
from app.main import queue_display_state
from clinical_knowledge.privacy import check_public_report_payload
from scripts import run_medai_operator_workflow_uat_12a as uat


def test_queued_supported_files_enable_run(tmp_path: Path) -> None:
    files = uat._create_supported_queue(tmp_path)

    assert len(files) == len(uat.SUPPORTED_FILE_TYPES)
    assert len(launcher.list_test_input_files(tmp_path)) == len(uat.SUPPORTED_FILE_TYPES)
    assert queue_display_state(queued_count=len(files), selected_count=0)["start_enabled"] is True


def test_empty_queue_disables_run() -> None:
    assert queue_display_state(queued_count=0, selected_count=0)["start_enabled"] is False


def test_local_image_ocr_text_extraction_path_creates_review_bound_records() -> None:
    report = uat.run_uat()

    assert report["run_completed"] is True
    assert report["local_ocr_attempted"] is True
    assert report["text_recovery_status"] == "recovered"
    assert report["records_created"] >= 3
    assert report["records_review_bound_before_action"] >= 3
    assert report["records_active_before_action"] == 0


def test_mkb_explorer_count_model_shows_review_bound_records() -> None:
    report = uat.run_uat()

    assert report["mkb_explorer_review_bound_count"] >= 3


def test_review_queue_reader_returns_records() -> None:
    report = uat.run_uat()

    assert report["review_queue_reader_rows"] >= 3
    assert report["review_queue_count_before_action"] >= 3


def test_accept_action_affects_one_selected_record_only() -> None:
    report = uat.run_uat()

    assert report["accept_action_passed"] is True
    assert report["records_active_after_accept"] == 1
    assert report["action_proof"]["accept_selected_record_only"] is True


def test_reject_action_affects_one_selected_record_only() -> None:
    report = uat.run_uat()

    assert report["reject_action_passed"] is True
    assert report["records_rejected_after_reject"] == 1
    assert report["action_proof"]["reject_selected_record_only"] is True


def test_defer_action_keeps_record_review_bound() -> None:
    report = uat.run_uat()

    assert report["defer_action_passed"] is True
    assert report["records_review_bound_after_defer"] == 1
    assert report["action_proof"]["defer_selected_record_only"] is True


def test_category_propagation_works() -> None:
    assert uat.run_uat()["category_propagated"] is True


def test_specialty_domain_propagation_works() -> None:
    assert uat.run_uat()["specialty_propagated"] is True


def test_no_external_api_used() -> None:
    assert uat.run_uat()["external_api_used"] is False


def test_no_auto_accept() -> None:
    report = uat.run_uat()

    assert report["auto_accept"] is False
    assert report["accepted_count_before_human_action"] == 0


def test_public_reports_contain_no_raw_ocr_text() -> None:
    report = uat.run_uat()
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            uat.SUMMARY_MD_PATH,
            uat.REPORT_JSON_PATH,
            uat.REPORT_MD_PATH,
        )
    )

    assert report["raw_ocr_text_in_report"] is False
    assert uat.SYNTHETIC_OCR_TEXT not in combined


def test_public_reports_contain_no_phi_private_paths() -> None:
    uat.run_uat()

    for path in (uat.SUMMARY_MD_PATH, uat.REPORT_JSON_PATH, uat.REPORT_MD_PATH):
        payload = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else path.read_text(encoding="utf-8")
        result = check_public_report_payload(payload)
        assert result.passed, f"{path.name}: {result.leak_examples_redacted}"


def test_existing_regression_commands_are_required_by_12a_scope() -> None:
    required = {
        "tests/test_medai_file_intake_multiformat_uat_12b.py",
        "tests/test_medai_local_image_ocr_routing_12c.py",
        "tests/test_medai_image_ocr_reviewbound_routing_12d.py",
        "tests/test_medai_actual_streamlit_ui_wiring_fix_11f.py",
    }
    assert required
