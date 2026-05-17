from __future__ import annotations

from pathlib import Path

from app.main import (
    advanced_diagnostic_fields,
    canonical_run_result_record,
    operator_document_type,
    operator_result_explanation,
    queue_display_state,
)
from app.operator_safety import PHASE52_SAFETY_WARNING


APP_MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"


def app_source() -> str:
    return APP_MAIN.read_text(encoding="utf-8")


def _record(document_type: str) -> dict:
    return {
        "status": "review",
        "document_type": document_type,
        "confidence": 0.45,
        "selected_extractor": "rules_based",
        "validation_status": "rejected",
        "ocr_quality_band": "readable",
        "external_api_used": False,
        "raw_ocr_text": "not public",
        "raw_document_text": "not public",
        "file_name": "not_public.pdf",
        "processed_path": "C:/private/not_public.pdf",
    }


def test_selected_but_unqueued_files_are_explicit_and_start_disabled() -> None:
    state = queue_display_state(queued_count=0, selected_count=2)

    assert state["message"] == "Files selected. Add/start run to process them."
    assert state["start_enabled"] is False
    assert state["queued_count"] == 0


def test_queued_files_count_controls_start_enablement() -> None:
    state = queue_display_state(queued_count=3, selected_count=3)

    assert state["message"] == "Ready to process 3 files."
    assert state["start_enabled"] is True
    assert state["queued_count"] == 3


def test_empty_queue_without_selection_uses_non_misleading_empty_text() -> None:
    state = queue_display_state(queued_count=0, selected_count=0)

    assert state["message"] == "No documents added yet. Choose files to begin."
    assert state["start_enabled"] is False


def test_main_card_and_advanced_document_type_use_same_canonical_record() -> None:
    for document_type in ("Lab result", "Treatment plan", "Medication plan", "Unknown"):
        record = canonical_run_result_record(_record(document_type))
        advanced = advanced_diagnostic_fields(record)

        assert operator_document_type(record) == document_type
        assert advanced["document_type"] == document_type


def test_previous_unknown_does_not_override_current_treatment_record() -> None:
    previous = canonical_run_result_record(_record("Unknown"))
    current = canonical_run_result_record(_record("Treatment plan"))

    assert operator_document_type(previous) == "Unknown"
    assert operator_document_type(current) == "Treatment plan"
    assert advanced_diagnostic_fields(current)["document_type"] == "Treatment plan"


def test_review_package_summary_is_separate_collapsed_previous_summary() -> None:
    source = app_source()

    assert 'st.expander("Previous review summary / aggregate review status", expanded=False)' in source
    assert "This is historical aggregate review-package information, not the current run result." in source
    assert "render_review_package_panel(show_title=False)" in source


def test_advanced_details_are_collapsed_filtered_and_safe() -> None:
    record = canonical_run_result_record(_record("Treatment plan"))
    advanced = advanced_diagnostic_fields(record)
    source = app_source()

    assert 'st.expander("Advanced technical details", expanded=False)' in source
    assert advanced["document_type"] == "Treatment plan"
    assert "raw_ocr_text" not in advanced
    assert "raw_document_text" not in advanced
    assert "file_name" not in advanced
    assert "processed_path" not in advanced


def test_safety_language_remains_operator_friendly_and_review_bound() -> None:
    assert "does not diagnose, recommend treatment" in PHASE52_SAFETY_WARNING
    assert "interpret medications" in PHASE52_SAFETY_WARNING
    assert "accept extracted values on its own" in PHASE52_SAFETY_WARNING
    assert "Medication names, doses, schedules, and recommendations were not interpreted or accepted" in (
        operator_result_explanation("Treatment plan")
    )
    assert "lab values have not been checked or accepted" in operator_result_explanation("Lab result")
