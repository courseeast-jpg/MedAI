from scripts.run_medai_ru_lab_extract_diag_01 import (
    cue_categories,
    likely_failure_reason,
    safe_text_visibility_diagnostic,
    table_like_pattern_detected,
)


def test_diagnostic_emits_only_buckets_and_categories_not_raw_text() -> None:
    text = "Лабораторное исследование\nГемоглобин 140 г/л норма"

    diagnostic = safe_text_visibility_diagnostic(
        safe_id="file_001",
        document_type_current="Unknown",
        extractor="spacy",
        confidence=0.63,
        ocr_quality="readable",
        text=text,
    )

    assert diagnostic["safe_id"] == "file_001"
    assert diagnostic["text_available"] is True
    assert "lab_header" in diagnostic["russian_lab_cue_categories_detected"]
    assert text not in str(diagnostic)
    assert "Гемоглобин" not in str(diagnostic)


def test_cyrillic_detection_is_bucketed_safely() -> None:
    diagnostic = safe_text_visibility_diagnostic(
        safe_id="file_001",
        document_type_current="Unknown",
        extractor="rules_based",
        confidence=0.45,
        ocr_quality="readable",
        text="Результаты показатель значение норма",
    )

    assert diagnostic["cyrillic_detected"] is True
    assert diagnostic["cyrillic_density_bucket"] in {"low", "medium", "high"}


def test_digit_and_table_like_detection_are_bucketed_safely() -> None:
    text = "Показатель    Значение    Норма\nГлюкоза    5.1    3.9-5.8\nАЛТ    20    0-40"

    diagnostic = safe_text_visibility_diagnostic(
        safe_id="file_001",
        document_type_current="Unknown",
        extractor="rules_based",
        confidence=0.45,
        ocr_quality="readable",
        text=text,
    )

    assert diagnostic["digit_density_bucket"] in {"low", "medium", "high"}
    assert diagnostic["table_like_pattern_detected"] is True
    assert diagnostic["line_count_bucket"] in {"few", "medium", "many"}
    assert table_like_pattern_detected(text) is True


def test_missing_text_is_reported_as_no_text_available() -> None:
    diagnostic = safe_text_visibility_diagnostic(
        safe_id="file_001",
        document_type_current="Unknown",
        extractor="spacy",
        confidence=None,
        ocr_quality="readable",
        text="",
    )

    assert diagnostic["text_available"] is False
    assert diagnostic["text_length_bucket"] == "none"
    assert diagnostic["likely_failure_reason"] == "no_text_available"


def test_sparse_text_is_reported_as_text_too_sparse() -> None:
    reason = likely_failure_reason(text="результат", categories=["result_terms"], current_document_type="Unknown")

    assert reason == "text_too_sparse"


def test_cue_categories_are_safe_labels_only() -> None:
    categories = cue_categories("Результаты показатель значение единицы норма креатинин")

    assert categories
    assert all(category.isascii() for category in categories)
    assert "analyte_terms" in categories


def test_no_clinical_interpretation_is_generated() -> None:
    diagnostic = safe_text_visibility_diagnostic(
        safe_id="file_001",
        document_type_current="Unknown",
        extractor="spacy",
        confidence=0.63,
        ocr_quality="readable",
        text="Глюкоза 5.1 ммоль/л норма",
    )

    combined = str(diagnostic).lower()
    assert "diagnosis" not in combined
    assert "dose" not in combined
    assert "recommend" not in combined


def test_external_api_remains_disabled_not_used() -> None:
    diagnostic = safe_text_visibility_diagnostic(
        safe_id="file_001",
        document_type_current="Unknown",
        extractor="rules_based",
        confidence=0.45,
        ocr_quality="readable",
        text="Результаты показатель значение единицы норма гемоглобин",
    )

    assert "external_api_used" not in diagnostic
    assert "cloud" not in str(diagnostic).lower()
