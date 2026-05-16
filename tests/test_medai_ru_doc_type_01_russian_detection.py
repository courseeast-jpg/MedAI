from __future__ import annotations

from app.lab_document_metadata import (
    LAB_RESULT_LABEL,
    MEDICATION_PLAN_LABEL,
    TREATMENT_PLAN_LABEL,
    UNKNOWN_DOCUMENT_LABEL,
    URINALYSIS_LABEL,
    classify_lab_document_type,
    review_reason_for_result,
)
from app.main import item_status, operator_review_reason_for_item


def test_synthetic_russian_lab_result_text_classifies_as_lab_result() -> None:
    text = """
    Лабораторное исследование
    Результаты анализ крови
    Показатель Значение Единицы Референсные значения
    Гемоглобин значение норма
    Лейкоциты значение норма
    """

    assert classify_lab_document_type(text) == LAB_RESULT_LABEL


def test_synthetic_russian_urinalysis_text_classifies_as_urinalysis() -> None:
    text = """
    Общий анализ мочи
    Моча удельный вес pH
    Белок глюкоза кетоны нитриты
    Лейкоциты эритроциты эпителий бактерии
    """

    assert classify_lab_document_type(text) == URINALYSIS_LABEL


def test_synthetic_russian_treatment_plan_text_classifies_as_treatment_plan() -> None:
    text = """
    План лечения
    Схема лечения
    Курс 10 дней
    Контроль состояния по назначению врача
    """

    assert classify_lab_document_type(text) == TREATMENT_PLAN_LABEL


def test_synthetic_russian_medication_plan_text_classifies_as_medication_plan() -> None:
    text = """
    Назначения
    Препарат принимать утром и вечером
    Дозировка мг
    Таблетка курс дней
    """

    assert classify_lab_document_type(text) == MEDICATION_PLAN_LABEL


def test_weak_generic_russian_text_remains_unknown() -> None:
    text = "Документ для справки. Информация получена и сохранена."

    assert classify_lab_document_type(text) == UNKNOWN_DOCUMENT_LABEL


def test_urinalysis_remains_more_specific_than_lab_result() -> None:
    text = """
    Результаты анализ мочи
    Показатель значение единицы норма
    Моча удельный вес белок лейкоциты эритроциты
    """

    assert classify_lab_document_type(text) == URINALYSIS_LABEL


def test_treatment_and_medication_labels_do_not_trigger_acceptance() -> None:
    treatment_item = {
        "status": "review",
        "validation_status": "rejected",
        "document_type": TREATMENT_PLAN_LABEL,
        "confidence": 0.45,
    }
    medication_item = {
        "status": "review",
        "validation_status": "rejected",
        "document_type": MEDICATION_PLAN_LABEL,
        "confidence": 0.45,
    }

    assert item_status(treatment_item) == "review"
    assert item_status(medication_item) == "review"


def test_document_type_detection_does_not_change_confidence() -> None:
    confidence = 0.45
    document_type = classify_lab_document_type("Результаты показатель значение единицы норма гемоглобин")

    assert document_type == LAB_RESULT_LABEL
    assert confidence == 0.45


def test_low_confidence_russian_lab_result_remains_needs_review() -> None:
    item = {
        "status": "review",
        "validation_status": "rejected",
        "document_type": LAB_RESULT_LABEL,
        "confidence": 0.45,
    }

    assert item_status(item) == "review"
    assert "confidence is below the acceptance gate" in operator_review_reason_for_item(item)


def test_treatment_and_medication_review_reasons_require_human_review() -> None:
    treatment_reason = review_reason_for_result(
        document_type=TREATMENT_PLAN_LABEL,
        validation_status="rejected",
        confidence=0.45,
        status="review",
    )
    medication_reason = review_reason_for_result(
        document_type=MEDICATION_PLAN_LABEL,
        validation_status="rejected",
        confidence=0.45,
        status="review",
    )

    assert treatment_reason == "Needs review: treatment-plan style document detected. Human review is required."
    assert medication_reason == "Needs review: medication-plan style document detected. Human review is required."


def test_no_clinical_interpretation_or_medication_advice_is_generated() -> None:
    reason = review_reason_for_result(
        document_type=MEDICATION_PLAN_LABEL,
        validation_status="rejected",
        confidence=0.45,
        status="review",
    )

    lower = reason.lower()
    assert "diagnosis" not in lower
    assert "normal" not in lower
    assert "abnormal" not in lower
    assert "take " not in lower
    assert "dose" not in lower


def test_external_api_remains_unused_by_metadata_detection() -> None:
    assert classify_lab_document_type("Показатель значение единицы референсные значения") == LAB_RESULT_LABEL
