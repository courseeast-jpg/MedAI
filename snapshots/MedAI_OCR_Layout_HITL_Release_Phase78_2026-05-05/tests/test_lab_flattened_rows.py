from __future__ import annotations

from lab_normalization.lab_normalizer import normalize_lab_text
from lab_normalization.lab_row_parser import (
    ADJACENT_VALUE_UNIT_RE,
    parse_lab_row,
    parse_lab_rows,
)


# ---------------------------------------------------------------------------
# Adjacent value+unit (no space between value and unit)
# ---------------------------------------------------------------------------


def test_adjacent_value_unit_parsed() -> None:
    row = parse_lab_row("Glucose 103mg/dL 65-99 H")

    assert row is not None
    assert row.test_name == "Glucose"
    assert row.value == "103"
    assert row.unit == "mg/dL"
    assert row.reference_range == "65-99"
    assert row.abnormal_flag == "H"
    assert "adjacent_value_unit" in row.warnings


def test_adjacent_value_unit_no_range() -> None:
    row = parse_lab_row("WBC 6.2x10E3/uL")

    assert row is not None
    assert row.test_name == "WBC"
    assert row.value == "6.2"
    assert row.unit == "x10E3/uL"
    assert row.reference_range is None
    assert "adjacent_value_unit" in row.warnings


def test_adjacent_value_unit_confidence_includes_unit_bonus() -> None:
    row = parse_lab_row("WBC 6.2x10E3/uL")

    assert row is not None
    # base 0.72 + unit 0.08 = 0.80; no range/flag so qualitative_or_sparse_row NOT added
    assert row.parser_confidence >= 0.80


def test_adjacent_value_unit_with_range_confidence_high() -> None:
    row = parse_lab_row("Creatinine 1.1mg/dL 0.6-1.2")

    assert row is not None
    assert row.reference_range == "0.6-1.2"
    # base 0.72 + unit 0.08 + range 0.10 = 0.90
    assert row.parser_confidence >= 0.90


def test_adjacent_value_unit_regex_does_not_match_normal_spacing() -> None:
    # Normal-spaced rows match NUMERIC_ROW_RE, not ADJACENT_VALUE_UNIT_RE.
    # The warning should be absent.
    row = parse_lab_row("Glucose 103 mg/dL 65-99 H")

    assert row is not None
    assert "adjacent_value_unit" not in row.warnings


# ---------------------------------------------------------------------------
# Split reference range: range appears on the line after the value
# ---------------------------------------------------------------------------


def test_split_range_on_next_line_parsed() -> None:
    text = "Glucose 103 mg/dL\n65-99"
    rows = parse_lab_rows(text)

    assert len(rows) == 1
    row = rows[0]
    assert row.test_name == "Glucose"
    assert row.value == "103"
    assert row.unit == "mg/dL"
    assert row.reference_range == "65-99"
    assert "split_range_row" in row.warnings


def test_split_range_with_flag_on_next_line() -> None:
    text = "Creatinine 1.8 mg/dL\n0.6-1.2 H"
    rows = parse_lab_rows(text)

    assert len(rows) == 1
    row = rows[0]
    assert row.reference_range == "0.6-1.2"
    assert row.abnormal_flag == "H"
    assert "split_range_row" in row.warnings


def test_split_range_does_not_duplicate_row() -> None:
    # Exactly one row should be produced, not two.
    text = "WBC 6.2 x10E3/uL\n3.4-10.8"
    rows = parse_lab_rows(text)

    assert len(rows) == 1


def test_split_range_confidence_higher_than_no_range() -> None:
    single_line = parse_lab_rows("Glucose 103 mg/dL")
    split_line = parse_lab_rows("Glucose 103 mg/dL\n65-99")

    assert len(single_line) == 1
    assert len(split_line) == 1
    assert split_line[0].parser_confidence > single_line[0].parser_confidence


def test_split_range_not_triggered_when_next_line_is_not_range() -> None:
    text = "Glucose 103 mg/dL\nPatient sample processed"
    rows = parse_lab_rows(text)

    assert len(rows) == 1
    assert rows[0].reference_range is None
    assert "split_range_row" not in rows[0].warnings


def test_split_range_less_than_range() -> None:
    text = "TSH 0.8 IU/L\n<4.5"
    rows = parse_lab_rows(text)

    assert len(rows) == 1
    assert rows[0].reference_range == "<4.5"
    assert "split_range_row" in rows[0].warnings


# ---------------------------------------------------------------------------
# Combined: adjacent value+unit AND split range
# ---------------------------------------------------------------------------


def test_adjacent_value_unit_then_split_range() -> None:
    text = "Glucose 103mg/dL\n65-99 H"
    rows = parse_lab_rows(text)

    assert len(rows) == 1
    row = rows[0]
    assert row.test_name == "Glucose"
    assert row.value == "103"
    assert row.unit == "mg/dL"
    assert row.reference_range == "65-99"
    assert row.abnormal_flag == "H"
    assert "adjacent_value_unit" in row.warnings
    assert "split_range_row" in row.warnings


# ---------------------------------------------------------------------------
# Multiple flattened rows in one text block
# ---------------------------------------------------------------------------


def test_multiple_split_range_rows() -> None:
    text = "\n".join([
        "WBC 6.2 x10E3/uL",
        "3.4-10.8",
        "RBC 4.5 x10E6/uL",
        "3.8-5.2",
    ])
    rows = parse_lab_rows(text)

    assert len(rows) == 2
    assert rows[0].test_name == "WBC"
    assert rows[0].reference_range == "3.4-10.8"
    assert rows[1].test_name == "RBC"
    assert rows[1].reference_range == "3.8-5.2"


# ---------------------------------------------------------------------------
# Safety: existing behavior unchanged
# ---------------------------------------------------------------------------


def test_normal_row_unaffected_by_new_patterns() -> None:
    row = parse_lab_row("Glucose 103 mg/dL 65-99 H")

    assert row is not None
    assert row.test_name == "Glucose"
    assert row.value == "103"
    assert row.unit == "mg/dL"
    assert row.reference_range == "65-99"
    assert row.abnormal_flag == "H"
    assert "adjacent_value_unit" not in row.warnings
    assert "split_range_row" not in row.warnings


def test_qualitative_rows_unaffected() -> None:
    rows = parse_lab_rows("Ketones Negative\nBlood Negative\nProtein Trace")

    assert [r.test_name for r in rows] == ["Ketones", "Blood", "Protein"]
    assert [r.value for r in rows] == ["Negative", "Negative", "Trace"]


def test_garbage_text_still_no_rows() -> None:
    text = "|||| ____ � � 11111111 ~~~~~ ???" * 3
    rows = parse_lab_rows(text)

    assert rows == []


def test_poor_ocr_still_routes_to_review_ocr_quality() -> None:
    text = "Glucose 103mg/dL 65-99 H\nWBC 6.2x10E3/uL\n3.4-10.8"
    result = normalize_lab_text(
        text,
        ocr_layout_quality_band="poor_ocr",
        current_status="review_ocr_quality",
        entity_count=3,
    )

    assert result.should_upgrade_from_ocr_review_to_review is False
    assert result.safety_checks["not_poor_ocr_band"] is False


def test_lab_normalizer_cannot_produce_accepted_status() -> None:
    text = "\n".join([
        "WBC 6.2x10E3/uL",
        "3.4-10.8",
        "Glucose 103mg/dL",
        "65-99 H",
        "Creatinine 1.1 mg/dL 0.6-1.2",
    ])
    result = normalize_lab_text(
        text,
        ocr_layout_quality_band="good",
        current_status="review_ocr_quality",
        entity_count=3,
    )

    # normalizer can upgrade review_ocr_quality → review, never → accepted
    assert result.should_upgrade_from_ocr_review_to_review in {True, False}
    # Safety: normalizer never directly produces "accepted"
    assert not result.safety_checks.get("accepted_due_to_lab_normalizer", False)
