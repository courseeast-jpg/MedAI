from scripts.run_medai_ru_lab_text_vis_01 import (
    has_table_like_numeric_text,
    normalize_visibility_text,
    safe_text_visibility_summary,
)


def test_safe_diagnostic_buckets_cyrillic_density_without_raw_text() -> None:
    text = "\u0413\u043b\u044e\u043a\u043e\u0437\u0430    5.1    3.9-5.8\n\u041a\u0440\u0435\u0430\u0442\u0438\u043d\u0438\u043d    80    60-110"

    diagnostic = safe_text_visibility_summary(
        safe_id="file_001",
        text=text,
        extractor_path_used="pymupdf_native_text",
        pdf_text_extraction_attempted=True,
        ocr_attempted=False,
        ocr_engine=None,
        ocr_language_config_bucket="english_only",
    )

    assert diagnostic["safe_id"] == "file_001"
    assert diagnostic["cyrillic_detected"] is True
    assert diagnostic["cyrillic_density_bucket"] in {"low", "medium", "high"}
    assert text not in str(diagnostic)
    assert "\u0413\u043b\u044e\u043a\u043e\u0437\u0430" not in str(diagnostic)


def test_unicode_cyrillic_text_is_preserved_by_visibility_normalization() -> None:
    normalized = normalize_visibility_text("\u0410\u043d\u0430\u043b\u0438\u0437 \u043c\u043e\u0447\u0438 \u0451")

    assert "\u0410\u043d\u0430\u043b\u0438\u0437" in normalized
    assert "\u043c\u043e\u0447\u0438" in normalized
    assert "\u0435" in normalized


def test_numeric_table_text_with_zero_cyrillic_is_flagged() -> None:
    text = "Result    Value    Range\nA1    5.1    3.9-5.8\nB2    80    60-110"

    diagnostic = safe_text_visibility_summary(
        safe_id="file_001",
        text=text,
        extractor_path_used="pymupdf_native_text",
        pdf_text_extraction_attempted=True,
        ocr_attempted=False,
        ocr_engine=None,
        ocr_language_config_bucket="english_only",
    )

    assert has_table_like_numeric_text(text) is True
    assert diagnostic["cyrillic_detected"] is False
    assert diagnostic["cyrillic_missing_despite_table_text"] is True
    assert diagnostic["likely_failure_reason"] == "ocr_skipped_due_to_numeric_readable_text"


def test_diagnostic_report_structure_contains_no_raw_text() -> None:
    text = "\u0410\u041b\u0422    20    0-40"

    diagnostic = safe_text_visibility_summary(
        safe_id="file_002",
        text=text,
        extractor_path_used="pymupdf_native_text",
        pdf_text_extraction_attempted=True,
        ocr_attempted=False,
        ocr_engine=None,
        ocr_language_config_bucket="multilingual",
    )

    assert "text" not in diagnostic
    assert text not in str(diagnostic)
    assert diagnostic["text_length_bucket"] in {"tiny", "short", "medium", "long"}


def test_no_confidence_score_changes_are_emitted() -> None:
    diagnostic = safe_text_visibility_summary(
        safe_id="file_001",
        text="Result    Value\nA1    5.1",
        extractor_path_used="pymupdf_native_text",
        pdf_text_extraction_attempted=True,
        ocr_attempted=False,
        ocr_engine=None,
        ocr_language_config_bucket="english_only",
    )

    assert "confidence" not in diagnostic
    assert "confidence_bucket" not in diagnostic


def test_no_auto_acceptance_fields_are_emitted() -> None:
    diagnostic = safe_text_visibility_summary(
        safe_id="file_001",
        text="Result    Value\nA1    5.1",
        extractor_path_used="pymupdf_native_text",
        pdf_text_extraction_attempted=True,
        ocr_attempted=False,
        ocr_engine=None,
        ocr_language_config_bucket="english_only",
    )

    assert "accepted" not in str(diagnostic).lower()
    assert "auto_accept" not in str(diagnostic).lower()


def test_no_external_api_use_is_emitted() -> None:
    diagnostic = safe_text_visibility_summary(
        safe_id="file_001",
        text="Result    Value\nA1    5.1",
        extractor_path_used="pymupdf_native_text",
        pdf_text_extraction_attempted=True,
        ocr_attempted=False,
        ocr_engine=None,
        ocr_language_config_bucket="english_only",
    )

    assert "external_api" not in str(diagnostic).lower()
    assert "cloud" not in str(diagnostic).lower()


def test_no_clinical_interpretation_is_generated() -> None:
    diagnostic = safe_text_visibility_summary(
        safe_id="file_001",
        text="\u0413\u043b\u044e\u043a\u043e\u0437\u0430    5.1    3.9-5.8",
        extractor_path_used="pymupdf_native_text",
        pdf_text_extraction_attempted=True,
        ocr_attempted=False,
        ocr_engine=None,
        ocr_language_config_bucket="multilingual",
    )

    combined = str(diagnostic).lower()
    assert "diagnosis" not in combined
    assert "dose" not in combined
    assert "recommend" not in combined
