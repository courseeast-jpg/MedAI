from app.lab_document_metadata import classify_lab_document_type
from ingestion.cyrillic_ocr_gate import (
    build_cyrillic_ocr_shadow_marker,
    cyrillic_ocr_shadow_gate_decision,
)

NUMERIC_TABLE_TEXT = "\n".join(
    [
        "Result    Value    Range",
        "A1    5.1    3.9-5.8",
        "B2    80    60-110",
        "C3    14.2    10-20",
        "D4    101    90-110",
        "E5    2.8    1.0-3.5",
        "F6    7.4    6.8-8.0",
        "G7    33    20-40",
        "H8    145    135-150",
        "I9    4.2    3.5-5.0",
        "J10    12    0-20",
    ]
) * 3


def test_numeric_table_readable_text_with_zero_cyrillic_triggers_shadow_recommendation() -> None:
    marker = cyrillic_ocr_shadow_gate_decision(
        text_length_bucket="long",
        digit_density_bucket="medium",
        cyrillic_density_bucket="none",
        table_like_pattern_detected=True,
        current_ocr_skipped=True,
        language_context="unknown",
    )

    assert marker["cyrillic_ocr_recommended"] is True
    assert marker["language_text_visibility"] == "incomplete"
    assert marker["ocr_gate_reason"] == "numeric_table_text_without_cyrillic"
    assert marker["review_only"] is True
    assert marker["auto_accept_allowed"] is False


def test_medium_digit_density_without_table_pattern_triggers_shadow_recommendation() -> None:
    marker = cyrillic_ocr_shadow_gate_decision(
        text_length_bucket="long",
        digit_density_bucket="medium",
        cyrillic_density_bucket="none",
        table_like_pattern_detected=False,
        current_ocr_skipped=True,
        language_context="unknown",
    )

    assert marker["cyrillic_ocr_recommended"] is True
    assert marker["language_text_visibility"] == "incomplete"
    assert marker["ocr_gate_reason"] == "numeric_table_text_without_cyrillic"


def test_cyrillic_visible_text_does_not_trigger_shadow_recommendation() -> None:
    marker = cyrillic_ocr_shadow_gate_decision(
        text_length_bucket="long",
        digit_density_bucket="medium",
        cyrillic_density_bucket="medium",
        table_like_pattern_detected=True,
        current_ocr_skipped=True,
    )

    assert marker["cyrillic_ocr_recommended"] is False
    assert marker["language_text_visibility"] == "visible"
    assert marker["ocr_gate_reason"] == "cyrillic_visible"


def test_no_text_documents_are_not_classified_as_this_gate() -> None:
    marker = cyrillic_ocr_shadow_gate_decision(
        text_length_bucket="none",
        digit_density_bucket="none",
        cyrillic_density_bucket="none",
        table_like_pattern_detected=False,
        current_ocr_skipped=True,
    )

    assert marker["cyrillic_ocr_recommended"] is False
    assert marker["ocr_gate_reason"] == "insufficient_native_text_for_shadow_gate"


def test_sparse_text_does_not_over_trigger() -> None:
    marker = cyrillic_ocr_shadow_gate_decision(
        text_length_bucket="short",
        digit_density_bucket="high",
        cyrillic_density_bucket="none",
        table_like_pattern_detected=True,
        current_ocr_skipped=True,
    )

    assert marker["cyrillic_ocr_recommended"] is False
    assert marker["language_text_visibility"] == "not_applicable"


def test_gate_marker_never_changes_confidence() -> None:
    marker = build_cyrillic_ocr_shadow_marker(
        NUMERIC_TABLE_TEXT,
        current_ocr_skipped=True,
    )

    assert "confidence" not in marker
    assert "threshold" not in marker


def test_gate_marker_never_changes_acceptance() -> None:
    marker = build_cyrillic_ocr_shadow_marker(
        NUMERIC_TABLE_TEXT,
        current_ocr_skipped=True,
    )

    assert marker["review_only"] is True
    assert marker["auto_accept_allowed"] is False
    assert "accepted" not in str(marker).lower()


def test_gate_marker_is_review_only() -> None:
    marker = build_cyrillic_ocr_shadow_marker(
        NUMERIC_TABLE_TEXT,
        current_ocr_skipped=True,
    )

    assert marker["cyrillic_ocr_recommended"] is True
    assert marker["review_only"] is True
    assert marker["ocr_fallback_executed"] is False


def test_gate_marker_produces_safe_metadata_only_no_raw_text() -> None:
    raw_text = NUMERIC_TABLE_TEXT
    marker = build_cyrillic_ocr_shadow_marker(raw_text, current_ocr_skipped=True)

    assert "text" not in marker
    assert raw_text not in str(marker)
    assert marker["text_length_bucket"] in {"medium", "long"}


def test_external_api_remains_disabled_not_used() -> None:
    marker = build_cyrillic_ocr_shadow_marker(
        NUMERIC_TABLE_TEXT,
        current_ocr_skipped=True,
    )

    assert "external_api" not in str(marker).lower()
    assert "cloud" not in str(marker).lower()


def test_existing_russian_document_type_detection_still_works() -> None:
    document_type = classify_lab_document_type(
        "\u041e\u0431\u0449\u0438\u0439 \u0430\u043d\u0430\u043b\u0438\u0437 \u043c\u043e\u0447\u0438\n"
        "\u0443\u0434\u0435\u043b\u044c\u043d\u044b\u0439 \u0432\u0435\u0441 pH \u0431\u0435\u043b\u043e\u043a \u043b\u0435\u0439\u043a\u043e\u0446\u0438\u0442\u044b"
    )

    assert document_type == "Urinalysis"
