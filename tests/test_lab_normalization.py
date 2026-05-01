from __future__ import annotations

from lab_normalization.lab_coverage import measure_lab_coverage
from lab_normalization.lab_normalizer import normalize_lab_text
from lab_normalization.lab_row_parser import parse_lab_row, parse_lab_rows
from lab_normalization.lab_table_detector import detect_lab_table


def test_lab_table_detection_on_clean_lab_text() -> None:
    text = "\n".join([
        "CBC Test Results",
        "WBC 6.2 x10E3/uL 3.4-10.8",
        "Glucose 103 mg/dL 65-99 H",
        "Specific Gravity 1.025 1.005-1.030",
    ])

    detection = detect_lab_table(text)

    assert detection.has_lab_table is True
    assert detection.table_confidence >= 0.45
    assert "cbc" in detection.detected_sections


def test_lab_row_parsing_numeric_result_unit_range() -> None:
    row = parse_lab_row("Glucose 103 mg/dL 65-99 H")

    assert row is not None
    assert row.test_name == "Glucose"
    assert row.value == "103"
    assert row.unit == "mg/dL"
    assert row.reference_range == "65-99"
    assert row.abnormal_flag == "H"


def test_lab_row_parsing_qualitative_result() -> None:
    rows = parse_lab_rows("Ketones Negative\nBlood Negative\nProtein Trace")

    assert [row.test_name for row in rows] == ["Ketones", "Blood", "Protein"]
    assert [row.value for row in rows] == ["Negative", "Negative", "Trace"]


def test_poor_garbage_text_does_not_produce_false_lab_table() -> None:
    text = "|||| ____ \ufffd \ufffd 11111111 ~~~~~ ???" * 3

    detection = detect_lab_table(text)
    rows = parse_lab_rows(text)

    assert detection.has_lab_table is False
    assert rows == []


def test_lab_coverage_scores_recovered_rows() -> None:
    text = "WBC 6.2 x10E3/uL 3.4-10.8\nGlucose 103 mg/dL 65-99 H\nKetones Negative"
    rows = parse_lab_rows(text)

    coverage = measure_lab_coverage(text, rows)

    assert coverage.parsed_lab_row_count == 3
    assert coverage.coverage_band in {"good", "partial"}
    assert "lab_table_recovered" in coverage.reason_codes


def test_lab_normalization_can_change_ocr_review_to_review() -> None:
    text = "\n".join([
        "Urinalysis Test Results",
        "Specific Gravity 1.025 1.005-1.030",
        "RBC UA 0-2 /hpf 0-2",
        "Ketones Negative",
        "Blood Negative",
        "Protein Trace",
    ])

    result = normalize_lab_text(
        text,
        ocr_layout_quality_band="good",
        current_status="review_ocr_quality",
        entity_count=3,
    )

    assert result.should_upgrade_from_ocr_review_to_review is True
    assert result.lab_table_detected is True
    assert result.coverage["parsed_lab_row_count"] >= 3


def test_lab_normalization_cannot_change_any_file_to_accepted() -> None:
    result = normalize_lab_text(
        "Glucose 103 mg/dL 65-99 H\nWBC 6.2 x10E3/uL 3.4-10.8\nKetones Negative",
        ocr_layout_quality_band="good",
        current_status="review_ocr_quality",
        entity_count=3,
    )

    assert result.should_upgrade_from_ocr_review_to_review is True


def test_poor_ocr_remains_review_ocr_quality() -> None:
    result = normalize_lab_text(
        "Glucose 103 mg/dL 65-99 H\nWBC 6.2 x10E3/uL 3.4-10.8\nKetones Negative",
        ocr_layout_quality_band="poor_ocr",
        current_status="review_ocr_quality",
        entity_count=3,
    )

    assert result.should_upgrade_from_ocr_review_to_review is False
    assert result.safety_checks["not_poor_ocr_band"] is False
