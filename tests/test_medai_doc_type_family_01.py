from __future__ import annotations

from pathlib import Path

from app.document_type_registry import (
    ADMINISTRATIVE_INSURANCE_LABEL,
    CLINICAL_NOTE_LABEL,
    DISCHARGE_SUMMARY_LABEL,
    IMAGING_REPORT_LABEL,
    LAB_RESULT_LABEL,
    MEDICATION_PLAN_LABEL,
    PATHOLOGY_REPORT_LABEL,
    PROCEDURE_REPORT_LABEL,
    REFERRAL_ORDER_LABEL,
    SUPPORTED_DOCUMENT_FAMILIES,
    SUPPORTED_LANGUAGE_PACKS,
    TREATMENT_PLAN_LABEL,
    UNKNOWN_DOCUMENT_LABEL,
    classify_document_family,
    document_family_classification_diagnostic,
)
from app.lab_document_metadata import (
    classify_lab_document_type,
    safe_fallback_ocr_classification_diagnostic,
)
from app.main import (
    item_status,
    operator_label_evidence,
    operator_result_explanation,
)
from ingestion.cyrillic_ocr_gate import build_cyrillic_ocr_shadow_marker, run_local_cyrillic_ocr_fallback


NUMERIC_NATIVE_TEXT = "\n".join(
    [
        "100 200 300 400 500 600 700 800 900",
        "1.1 2.2 3.3 4.4 5.5 6.6 7.7 8.8",
        "10 20 30 40 50 60 70 80 90",
    ]
    * 40
)

RUSSIAN_MRI_REPORT = """
\u041c\u0420\u0422
\u0410\u043f\u043f\u0430\u0440\u0430\u0442
\u041c\u0420-\u0442\u043e\u043c\u043e\u0433\u0440\u0430\u043c\u043c\u044b
\u041e\u043f\u0438\u0441\u0430\u043d\u0438\u0435
\u0417\u0430\u043a\u043b\u044e\u0447\u0435\u043d\u0438\u0435
"""

ENGLISH_IMAGING_REPORT = """
MRI scanner
Description
Series and sequence
Impression
"""

POLISH_IMAGING_REPORT = """
Rezonans magnetyczny
Aparat
Opis
Wniosek
"""

ALBANIAN_IMAGING_REPORT = """
Rezonance magnetike
Aparat
Pershkrim
Perfundim
"""

RUSSIAN_LAB_TEXT = """
\u0411\u0438\u043e\u043c\u0430\u0442\u0435\u0440\u0438\u0430\u043b
\u0420\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442
\u0420\u0435\u0444\u0435\u0440\u0435\u043d\u0441\u043d\u044b\u0435 \u0437\u043d\u0430\u0447\u0435\u043d\u0438\u044f
\u041f\u043e\u043a\u0430\u0437\u0430\u0442\u0435\u043b\u044c
\u0413\u043b\u044e\u043a\u043e\u0437\u0430
"""

RUSSIAN_TREATMENT_TEXT = """
\u041f\u043b\u0430\u043d \u043b\u0435\u0447\u0435\u043d\u0438\u044f
\u0414\u0430\u0442\u0430 01.01 02.01 03.01
\u0424\u0438\u0437\u0438\u043e\u043f\u0440\u043e\u0446\u0435\u0434\u0443\u0440\u044b
\u0420\u0435\u0436\u0438\u043c \u0438 \u0440\u0435\u043a\u043e\u043c\u0435\u043d\u0434\u0430\u0446\u0438\u0438
"""

NON_MEDICAL_CYRILLIC_TEXT = "\u0417\u0430\u043c\u0435\u0442\u043a\u0438 \u0434\u043b\u044f \u0432\u0441\u0442\u0440\u0435\u0447\u0438 \u0438 \u043f\u043b\u0430\u043d \u043f\u0440\u043e\u0435\u043a\u0442\u0430."

AMBIGUOUS_MIXED_TEXT = """
MRI scanner
Description
Treatment plan
Schedule
Date
"""


def _recommended_marker() -> dict:
    return build_cyrillic_ocr_shadow_marker(NUMERIC_NATIVE_TEXT, current_ocr_skipped=True)


def test_registry_exposes_requested_families_and_language_packs() -> None:
    for family in (
        LAB_RESULT_LABEL,
        IMAGING_REPORT_LABEL,
        TREATMENT_PLAN_LABEL,
        MEDICATION_PLAN_LABEL,
        CLINICAL_NOTE_LABEL,
        DISCHARGE_SUMMARY_LABEL,
        REFERRAL_ORDER_LABEL,
        PROCEDURE_REPORT_LABEL,
        PATHOLOGY_REPORT_LABEL,
        ADMINISTRATIVE_INSURANCE_LABEL,
        UNKNOWN_DOCUMENT_LABEL,
    ):
        assert family in SUPPORTED_DOCUMENT_FAMILIES

    assert SUPPORTED_LANGUAGE_PACKS == ("english", "russian", "polish", "albanian")


def test_multilingual_imaging_reports_classify_as_imaging_report() -> None:
    for text in (
        RUSSIAN_MRI_REPORT,
        ENGLISH_IMAGING_REPORT,
        POLISH_IMAGING_REPORT,
        ALBANIAN_IMAGING_REPORT,
    ):
        diagnostic = document_family_classification_diagnostic(text)

        assert classify_document_family(text) == IMAGING_REPORT_LABEL
        assert diagnostic["candidate_family"] == IMAGING_REPORT_LABEL
        assert "imaging_modality" in diagnostic["matched_family_cue_keys"]
        assert diagnostic["review_only"] is True
        assert diagnostic["auto_accept_allowed"] is False


def test_existing_russian_lab_and_treatment_behavior_is_preserved() -> None:
    assert classify_lab_document_type(RUSSIAN_LAB_TEXT) == LAB_RESULT_LABEL
    assert classify_lab_document_type(RUSSIAN_TREATMENT_TEXT) == TREATMENT_PLAN_LABEL
    assert classify_document_family(RUSSIAN_LAB_TEXT) == LAB_RESULT_LABEL


def test_non_medical_cyrillic_and_ambiguous_text_remain_unknown() -> None:
    non_medical = document_family_classification_diagnostic(NON_MEDICAL_CYRILLIC_TEXT)
    ambiguous = document_family_classification_diagnostic(AMBIGUOUS_MIXED_TEXT)

    assert classify_document_family(NON_MEDICAL_CYRILLIC_TEXT) == UNKNOWN_DOCUMENT_LABEL
    assert non_medical["classification_block_reason"] == "too_few_safe_family_cue_keys"
    assert classify_document_family(AMBIGUOUS_MIXED_TEXT) == UNKNOWN_DOCUMENT_LABEL
    assert ambiguous["classification_block_reason"] == "ambiguous_family_candidates"
    assert ambiguous["ambiguous_candidates"]


def test_fallback_ocr_diagnostic_includes_safe_family_metadata_only() -> None:
    diagnostic = safe_fallback_ocr_classification_diagnostic(RUSSIAN_MRI_REPORT)
    family = diagnostic["document_family_classification_diagnostic"]

    assert diagnostic["matched_document_type_candidate"] == IMAGING_REPORT_LABEL
    assert family["candidate_family"] == IMAGING_REPORT_LABEL
    assert family["matched_family_cue_keys"]
    assert family["matched_language_cue_groups"]
    assert "raw_text" not in diagnostic
    assert "ocr_text" not in diagnostic
    assert RUSSIAN_MRI_REPORT not in str(diagnostic)


def test_fallback_family_classification_remains_review_only_and_local(tmp_path: Path) -> None:
    metadata = run_local_cyrillic_ocr_fallback(
        tmp_path / "synthetic.pdf",
        _recommended_marker(),
        language_lister=lambda: ["rus", "eng"],
        ocr_runner=lambda _path, _language: RUSSIAN_MRI_REPORT,
        document_type_classifier=classify_lab_document_type,
        classification_diagnostic_builder=safe_fallback_ocr_classification_diagnostic,
    )
    item = {
        "status": "review",
        "validation_status": "rejected",
        "document_type": metadata["ocr_gate_fallback_document_type"],
        "external_api_used": False,
    }

    assert metadata["ocr_gate_fallback_document_type"] == IMAGING_REPORT_LABEL
    assert item_status(item) == "review"
    assert metadata["ocr_gate_fallback_review_only"] is True
    assert metadata["ocr_gate_fallback_auto_accept_allowed"] is False
    assert "confidence" not in metadata
    assert "external_api" not in str(metadata).lower()
    assert "cloud" not in str(metadata).lower()
    assert RUSSIAN_MRI_REPORT not in str(metadata)


def test_operator_wording_supports_imaging_clinical_note_and_discharge_summary() -> None:
    imaging = operator_result_explanation(IMAGING_REPORT_LABEL)
    clinical = operator_result_explanation(CLINICAL_NOTE_LABEL)
    discharge = operator_result_explanation(DISCHARGE_SUMMARY_LABEL)

    assert "imaging-report style document" in imaging
    assert "Imaging findings and conclusions were not interpreted or accepted" in imaging
    assert "clinical-note style document" in clinical
    assert "Medical meaning was not interpreted or accepted" in clinical
    assert "discharge-summary style document" in discharge
    assert "Diagnoses, medications, and recommendations were not interpreted or accepted" in discharge
    assert operator_label_evidence(IMAGING_REPORT_LABEL) == [
        "Imaging modality wording found",
        "Description/conclusion structure found",
        "Imaging-report layout found",
    ]
