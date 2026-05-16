from __future__ import annotations

from app.lab_document_metadata import (
    LAB_RESULT_LABEL,
    UNKNOWN_DOCUMENT_LABEL,
    URINALYSIS_LABEL,
    classify_lab_document_type,
    review_reason_for_result,
)
from app.main import item_status, operator_review_reason_for_item


def test_general_lab_result_text_with_multiple_cues_is_lab_result() -> None:
    text = """
    Test results
    Specimen type: blood
    Collected: synthetic date
    Reported: synthetic date
    Component Result Value Units Reference Interval Flag
    """

    assert classify_lab_document_type(text) == LAB_RESULT_LABEL


def test_cbc_cmp_style_text_is_lab_result() -> None:
    text = """
    CBC and comprehensive metabolic panel
    WBC result unit reference range
    RBC result unit reference range
    Hemoglobin value units
    Platelet value units
    Glucose value units
    Creatinine value units
    """

    assert classify_lab_document_type(text) == LAB_RESULT_LABEL


def test_lipid_glucose_creatinine_reference_interval_text_is_lab_result() -> None:
    text = """
    Lipid panel
    Cholesterol result units reference interval
    Triglycerides value units
    HDL value
    LDL value
    Glucose result value
    Creatinine result value
    """

    assert classify_lab_document_type(text) == LAB_RESULT_LABEL


def test_urinalysis_remains_more_specific_than_lab_result() -> None:
    text = """
    Urinalysis
    Urine specific gravity result
    pH value
    Protein result
    RBC value
    WBC value
    Reference interval
    """

    assert classify_lab_document_type(text) == URINALYSIS_LABEL


def test_single_weak_result_word_remains_unknown() -> None:
    assert classify_lab_document_type("The project result was reviewed by the operator.") == UNKNOWN_DOCUMENT_LABEL


def test_irrelevant_non_medical_text_remains_unknown() -> None:
    text = "Invoice summary, delivery status, and customer service notes."

    assert classify_lab_document_type(text) == UNKNOWN_DOCUMENT_LABEL


def test_lab_result_detection_does_not_change_confidence() -> None:
    confidence = 0.45
    document_type = classify_lab_document_type("Test result component value units reference interval glucose")

    assert document_type == LAB_RESULT_LABEL
    assert confidence == 0.45


def test_lab_result_detection_does_not_make_item_accepted() -> None:
    item = {
        "status": "review",
        "validation_status": "rejected",
        "document_type": LAB_RESULT_LABEL,
        "confidence": 0.45,
    }

    assert item_status(item) == "review"


def test_low_confidence_lab_result_maps_to_needs_review_reason() -> None:
    item = {
        "status": "review",
        "validation_status": "rejected",
        "document_type": LAB_RESULT_LABEL,
        "confidence": 0.45,
    }

    reason = operator_review_reason_for_item(item)

    assert item_status(item) == "review"
    assert "lab-style document detected" in reason
    assert "confidence is below the acceptance gate" in reason


def test_review_reason_for_low_confidence_lab_result_is_clear() -> None:
    reason = review_reason_for_result(
        document_type=LAB_RESULT_LABEL,
        validation_status="rejected",
        confidence=0.45,
        status="review",
    )

    assert "lab-style document detected" in reason
    assert "confidence is below the acceptance gate" in reason


def test_external_api_and_clinical_interpretation_are_not_added() -> None:
    reason = review_reason_for_result(
        document_type=LAB_RESULT_LABEL,
        validation_status="rejected",
        confidence=0.45,
        status="review",
    )

    assert "diagnosis" not in reason.lower()
    assert "normal" not in reason.lower()
    assert "abnormal" not in reason.lower()
    assert "treat" not in reason.lower()
