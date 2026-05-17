from __future__ import annotations

from pathlib import Path

from app.main import (
    advanced_diagnostic_fields,
    medai_did_not_do_checklist,
    next_actions_for_document_type,
    operator_label_evidence,
    operator_result_explanation,
    russian_text_recovery_summary,
    text_recovery_chip,
)
from app.operator_safety import PHASE52_SAFETY_WARNING


APP_MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"


def app_source() -> str:
    return APP_MAIN.read_text(encoding="utf-8")


def test_lab_result_uses_plain_operator_wording() -> None:
    text = operator_result_explanation("Lab result")

    assert "lab-style document" in text
    assert "recovering readable Russian text locally" in text
    assert "lab values have not been checked or accepted" in text
    assert "compare the result with the source PDF" in text


def test_treatment_and_medication_plans_use_plain_operator_wording() -> None:
    treatment = operator_result_explanation("Treatment plan")
    medication = operator_result_explanation("Medication plan")

    for text in (treatment, medication):
        assert "recovering readable Russian text locally" in text
        assert "Medication names, doses, schedules, and recommendations were not interpreted or accepted" in text
        assert "A human must review the source PDF" in text


def test_unknown_document_uses_plain_operator_wording() -> None:
    assert operator_result_explanation("Unknown") == (
        "MedAI could not confidently identify this document type. A human must review the source PDF."
    )


def test_plain_evidence_hides_raw_cue_keys_from_main_ui() -> None:
    lab_cues = operator_label_evidence("Lab result")
    treatment_cues = operator_label_evidence("Treatment plan")
    unknown_cues = operator_label_evidence("Unknown")

    assert lab_cues == [
        "Biomaterial / result wording found",
        "Report and table structure found",
        "Lab-style layout found",
    ]
    assert treatment_cues == [
        "Treatment or recommendation section found",
        "Schedule-style layout found",
        "Date/grid pattern found",
    ]
    assert unknown_cues == ["No sufficient document-format clues matched"]
    for cue in lab_cues + treatment_cues + unknown_cues:
        assert "_" not in cue


def test_summary_chips_and_text_recovery_are_operator_friendly() -> None:
    recovered = {
        "ocr_gate_fallback_executed": True,
        "ocr_gate_fallback_cyrillic_detected": True,
        "ocr_gate_fallback_text_visibility": "recovered",
        "external_api_used": False,
    }
    not_needed = {"cyrillic_ocr_recommended": False, "ocr_gate_fallback_executed": False}

    assert text_recovery_chip(recovered) == "Worked"
    assert text_recovery_chip(not_needed) == "Not needed"
    assert russian_text_recovery_summary(recovered) == {
        "Russian text recovered": "Yes",
        "Local tool used": "Yes",
        "Cloud tools used": "No",
        "Human review still required": "Yes",
    }


def test_next_actions_and_safety_checklist_block_auto_acceptance_wording() -> None:
    assert next_actions_for_document_type("Lab result") == [
        "Open the source PDF.",
        "Compare each visible value with the source document.",
        "Mark anything uncertain.",
        "Sign off only after manual review.",
    ]
    assert "Do not rely on MedAI for medication names, dose, schedule, or recommendations." in next_actions_for_document_type(
        "Treatment plan"
    )
    checklist = medai_did_not_do_checklist()
    assert "Did not diagnose anything." in checklist
    assert "Did not accept lab values." in checklist
    assert "Did not send data to the cloud." in checklist


def test_advanced_technical_details_are_filtered_and_collapsed() -> None:
    item = {
        "document_type": "Lab result",
        "confidence": 0.45,
        "selected_extractor": "rules_based",
        "ocr_gate_fallback_engine": "tesseract_local",
        "raw_ocr_text": "must not appear",
        "raw_document_text": "must not appear",
        "file_name": "must_not_appear.pdf",
        "private_path": "C:/private/must/not/appear.pdf",
    }

    advanced = advanced_diagnostic_fields(item)
    source = app_source()

    assert advanced["document_type"] == "Lab result"
    assert advanced["selected_extractor"] == "rules_based"
    assert "raw_ocr_text" not in advanced
    assert "raw_document_text" not in advanced
    assert "file_name" not in advanced
    assert "private_path" not in advanced
    assert 'st.expander("Advanced technical details", expanded=False)' in source
    assert "Show raw run record" not in source


def test_operator_section_labels_replace_technical_section_labels() -> None:
    source = app_source()

    for text in (
        "Why MedAI labeled it this way",
        "Russian text recovery",
        "What happened",
        "What you need to do next",
        "What MedAI did not do",
        "Advanced technical details",
        "Cloud tools",
        "Acceptance",
    ):
        assert text in source

    for old_label in (
        "Document Type Evidence",
        "OCR / Cyrillic Recovery",
        "File processing timeline",
        "Operator Next Action",
        "Advanced diagnostics",
    ):
        assert old_label not in source


def test_warning_banner_uses_operator_safety_language() -> None:
    assert PHASE52_SAFETY_WARNING == (
        "Review required. Not for diagnosis. MedAI does not diagnose, recommend treatment, "
        "interpret medications, or accept extracted values on its own."
    )
