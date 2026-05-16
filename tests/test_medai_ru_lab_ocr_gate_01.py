from scripts.run_medai_ru_lab_ocr_gate_01 import (
    safe_ocr_gate_decision,
    synthetic_cyrillic_ocr_probe,
)


def test_ocr_gate_diagnostic_uses_metadata_buckets_only() -> None:
    diagnostic = safe_ocr_gate_decision(
        native_text_length_bucket="long",
        digit_density_bucket="medium",
        cyrillic_density_bucket="none",
        table_like_pattern_detected=True,
        current_ocr_skipped=True,
    )

    assert diagnostic["native_text_length_bucket"] == "long"
    assert diagnostic["digit_density_bucket"] == "medium"
    assert diagnostic["cyrillic_density_bucket"] == "none"
    assert "text" not in diagnostic
    assert "raw" not in str(diagnostic).lower()


def test_numeric_table_readable_zero_cyrillic_recommends_gate_not_acceptance() -> None:
    diagnostic = safe_ocr_gate_decision(
        native_text_length_bucket="long",
        digit_density_bucket="high",
        cyrillic_density_bucket="none",
        table_like_pattern_detected=True,
        current_ocr_skipped=True,
    )

    assert diagnostic["cyrillic_visibility_ocr_gate_needed"] is True
    assert diagnostic["proposed_gate_reason"] == "native_numeric_table_text_without_cyrillic"
    assert diagnostic["safe_mode"] == "review_only"
    assert diagnostic["auto_acceptance_allowed"] is False


def test_normal_cyrillic_visible_text_does_not_require_gate() -> None:
    diagnostic = safe_ocr_gate_decision(
        native_text_length_bucket="long",
        digit_density_bucket="medium",
        cyrillic_density_bucket="medium",
        table_like_pattern_detected=True,
        current_ocr_skipped=True,
    )

    assert diagnostic["cyrillic_visibility_ocr_gate_needed"] is False
    assert diagnostic["proposed_gate_reason"] == "language_text_visible"


def test_empty_no_text_documents_remain_separate_from_numeric_only_documents() -> None:
    diagnostic = safe_ocr_gate_decision(
        native_text_length_bucket="none",
        digit_density_bucket="none",
        cyrillic_density_bucket="none",
        table_like_pattern_detected=False,
        current_ocr_skipped=True,
    )

    assert diagnostic["cyrillic_visibility_ocr_gate_needed"] is False
    assert diagnostic["proposed_gate_reason"] == "native_text_not_substantial"


def test_proposed_gate_never_changes_confidence() -> None:
    diagnostic = safe_ocr_gate_decision(
        native_text_length_bucket="medium",
        digit_density_bucket="medium",
        cyrillic_density_bucket="none",
        table_like_pattern_detected=True,
        current_ocr_skipped=True,
    )

    assert "confidence" not in diagnostic
    assert "threshold" not in diagnostic


def test_proposed_gate_never_changes_auto_acceptance() -> None:
    diagnostic = safe_ocr_gate_decision(
        native_text_length_bucket="medium",
        digit_density_bucket="medium",
        cyrillic_density_bucket="none",
        table_like_pattern_detected=True,
        current_ocr_skipped=True,
    )

    assert diagnostic["auto_acceptance_allowed"] is False
    assert "accepted" not in str(diagnostic).lower()


def test_synthetic_probe_does_not_record_raw_ocr_text_when_unavailable() -> None:
    probe = synthetic_cyrillic_ocr_probe(tesseract_available=False, languages=set())

    assert probe["attempted"] is False
    assert probe["cyrillic_detected_bucket"] == "unavailable"
    assert probe["raw_text_recorded"] is False
    assert "text" not in probe


def test_external_api_remains_disabled_not_used() -> None:
    diagnostic = safe_ocr_gate_decision(
        native_text_length_bucket="long",
        digit_density_bucket="medium",
        cyrillic_density_bucket="none",
        table_like_pattern_detected=True,
        current_ocr_skipped=True,
    )

    assert "external_api" not in str(diagnostic).lower()
    assert "cloud" not in str(diagnostic).lower()
