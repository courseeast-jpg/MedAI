from app.lab_document_metadata import (
    LAB_RESULT_LABEL,
    MEDICATION_PLAN_LABEL,
    TREATMENT_PLAN_LABEL,
    UNKNOWN_DOCUMENT_LABEL,
    URINALYSIS_LABEL,
    classify_lab_document_type,
    display_document_type,
    review_reason_for_result,
    safe_document_type_diagnostic,
)
from app.main import item_status


def test_safe_diagnostic_emits_cue_categories_not_raw_text() -> None:
    text = "Результаты анализ крови показатель значение единицы норма гемоглобин"

    diagnostic = safe_document_type_diagnostic(text=text, document_type_before="unknown")

    assert diagnostic["document_type_after"] == LAB_RESULT_LABEL
    assert "result_terms" in diagnostic["russian_lab_cue_categories_detected"]
    assert text not in str(diagnostic)
    assert "гемоглобин" not in str(diagnostic)


def test_cyrillic_density_is_bucketed_not_printed_as_text() -> None:
    diagnostic = safe_document_type_diagnostic(text="Результаты показатель значение")

    assert diagnostic["cyrillic_detected"] is True
    assert diagnostic["cyrillic_density_bucket"] in {"low", "medium", "high"}
    assert "Результаты" not in str(diagnostic)


def test_missing_classifier_input_is_detected_safely() -> None:
    diagnostic = safe_document_type_diagnostic(text="", document_type_before="unknown")

    assert diagnostic["text_available"] is False
    assert diagnostic["text_length_bucket"] == "none"
    assert diagnostic["likely_reason_unknown"] == "no_text_available"


def test_unknown_existing_metadata_does_not_block_available_lab_text() -> None:
    text = "Лабораторное исследование результаты показатель значение единицы норма креатинин"

    assert display_document_type("unknown", text=text) == LAB_RESULT_LABEL


def test_multiple_russian_lab_cue_categories_support_lab_result() -> None:
    text = "Общий анализ крови результаты показатель значение единицы норма лейкоциты"
    diagnostic = safe_document_type_diagnostic(text=text, document_type_before="generic")

    assert classify_lab_document_type(text) == LAB_RESULT_LABEL
    assert diagnostic["document_type_after"] == LAB_RESULT_LABEL
    assert diagnostic["cue_category_count"] >= 3


def test_sparse_cyrillic_text_remains_unknown() -> None:
    text = "Пациент принес документы на прием"

    assert classify_lab_document_type(text) == UNKNOWN_DOCUMENT_LABEL
    diagnostic = safe_document_type_diagnostic(text=text)
    assert diagnostic["document_type_after"] == UNKNOWN_DOCUMENT_LABEL
    assert diagnostic["likely_reason_unknown"] == "too_few_cue_categories"


def test_urinalysis_remains_urinalysis() -> None:
    text = "Общий анализ мочи моча удельный вес белок глюкоза лейкоциты эритроциты"

    assert display_document_type("unknown", text=text) == URINALYSIS_LABEL


def test_treatment_and_medication_labels_remain_review_only() -> None:
    treatment = {
        "status": "review",
        "validation_status": "needs_review",
        "document_type": TREATMENT_PLAN_LABEL,
        "confidence": 0.9,
    }
    medication = {
        "status": "review",
        "validation_status": "needs_review",
        "document_type": MEDICATION_PLAN_LABEL,
        "confidence": 0.9,
    }

    assert item_status(treatment) == "review"
    assert item_status(medication) == "review"
    assert "Human review is required" in review_reason_for_result(
        document_type=TREATMENT_PLAN_LABEL,
        validation_status="needs_review",
        confidence=0.9,
        status="review",
    )


def test_detection_does_not_change_confidence_or_acceptance() -> None:
    confidence = 0.45
    item = {
        "status": "review",
        "validation_status": "rejected",
        "document_type": display_document_type(
            "unknown",
            text="Результаты анализ крови показатель значение единицы норма гемоглобин",
        ),
        "confidence": confidence,
    }

    assert item["document_type"] == LAB_RESULT_LABEL
    assert item["confidence"] == confidence
    assert item_status(item) == "review"


def test_external_api_and_clinical_interpretation_are_not_generated() -> None:
    reason = review_reason_for_result(
        document_type=LAB_RESULT_LABEL,
        validation_status="rejected",
        confidence=0.45,
        status="review",
    )

    assert "diagnosis" not in reason.lower()
    assert "dose" not in reason.lower()
    assert "external" not in reason.lower()
    assert "lab-style document detected" in reason
