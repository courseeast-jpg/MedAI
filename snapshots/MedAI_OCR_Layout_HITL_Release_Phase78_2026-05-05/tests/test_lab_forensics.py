from __future__ import annotations

from lab_normalization.lab_row_parser import debug_parse_lab_lines, parse_lab_rows


def _reasons(debug_output: dict) -> list[str]:
    return [r["reason"] for r in debug_output["rejected_lines"]]


# ---------------------------------------------------------------------------
# Rejection reason classification
# ---------------------------------------------------------------------------


def test_range_only_line_classified() -> None:
    debug = debug_parse_lab_lines("65-99")

    assert debug["candidate_line_count"] == 1
    assert debug["parsed_row_count"] == 0
    assert "range_only" in _reasons(debug)


def test_name_only_line_classified() -> None:
    debug = debug_parse_lab_lines("Glucose")

    assert "name_only" in _reasons(debug)


def test_value_only_line_classified() -> None:
    debug = debug_parse_lab_lines("103.5")

    assert "value_only" in _reasons(debug)


def test_empty_or_noise_line_classified() -> None:
    # Pure punctuation that survives candidate filtering (length >= 3)
    debug = debug_parse_lab_lines("...")

    # Either filtered out entirely (empty after normalize) or classified as empty/noise
    assert debug["parsed_row_count"] == 0


def test_malformed_adjacent_row_classified() -> None:
    # name + digits but no recognized unit and no valid range structure
    debug = debug_parse_lab_lines("Glucose 103 xyz")

    assert debug["parsed_row_count"] == 0
    reasons = _reasons(debug)
    assert any(r in reasons for r in ("no_unit_or_qualitative", "malformed_lab_row", "unrecognized"))


def test_qualitative_only_line_classified() -> None:
    debug = debug_parse_lab_lines("Negative")

    assert "qualitative_only" in _reasons(debug)


def test_signal_summary_detects_split_rows() -> None:
    text = "\n".join([
        "Glucose",
        "103",
        "65-99",
    ])
    debug = debug_parse_lab_lines(text)

    summary = debug["signal_summary"]
    assert summary["has_name_only_lines"] is True
    assert summary["has_value_only_lines"] is True
    assert summary["has_range_only_lines"] is True


def test_signal_summary_text_too_sparse() -> None:
    debug = debug_parse_lab_lines("Glucose")

    assert debug["signal_summary"]["text_too_sparse"] is True


# ---------------------------------------------------------------------------
# Debug mode does not change production behavior
# ---------------------------------------------------------------------------


def test_debug_does_not_mutate_normal_parse_output() -> None:
    text = "\n".join([
        "Glucose 103 mg/dL 65-99 H",
        "WBC 6.2 x10E3/uL 3.4-10.8",
        "Ketones Negative",
    ])

    rows_before = parse_lab_rows(text)
    debug = debug_parse_lab_lines(text)
    rows_after = parse_lab_rows(text)

    assert [r.to_dict() for r in rows_before] == [r.to_dict() for r in rows_after]
    assert debug["parsed_row_count"] == len(rows_before)


def test_debug_parsed_rows_match_production_parser() -> None:
    text = "Glucose 103mg/dL 65-99 H"

    debug = debug_parse_lab_lines(text)
    rows = parse_lab_rows(text)

    assert debug["parsed_row_count"] == len(rows) == 1
    assert debug["parsed_rows"][0]["row"]["test_name"] == "Glucose"


def test_debug_caps_output_at_50_entries() -> None:
    # Construct 80 unparseable lines
    text = "\n".join(f"junk_{i:03d}" for i in range(80))

    debug = debug_parse_lab_lines(text)

    assert len(debug["candidate_lines"]) <= 50
    assert len(debug["rejected_lines"]) <= 50
    # but the count fields reflect the true totals
    assert debug["candidate_line_count"] >= 50
