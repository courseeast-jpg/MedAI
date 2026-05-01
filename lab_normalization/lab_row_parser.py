from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


QUALITATIVE_VALUES = r"Negative|Positive|Trace|Not detected|None detected|Detected|Normal|Abnormal"
UNIT_PATTERN = r"(?:mg/dL|g/dL|x10E3/uL|x10E6/uL|x10E6/mL|/hpf|/HPF|mmol/L|mL/min/1\.73|CFU/mL|IU/L|U/L|%|Lg)"
RANGE_PATTERN = r"(?:[<>]?\d+(?:\.\d+)?\s*-\s*[<>]?\d+(?:\.\d+)?|[<>]\s*\d+(?:\.\d+)?)"
ABNORMAL_PATTERN = r"(?:H|L|High|Low|Abnormal)"

NUMERIC_ROW_RE = re.compile(
    rf"^\s*(?P<name>[A-Za-z][A-Za-z0-9 /().,%+-]{{1,60}}?)\s+"
    rf"(?P<value>[<>]?\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?|10\^\d+(?:\.\d+)?)"
    rf"(?:\s+(?P<unit>{UNIT_PATTERN}))?"
    rf"(?:\s+(?P<range>{RANGE_PATTERN}))?"
    rf"(?:\s+(?P<flag>{ABNORMAL_PATTERN}))?\s*$",
    re.I,
)

QUALITATIVE_ROW_RE = re.compile(
    rf"^\s*(?P<name>[A-Za-z][A-Za-z0-9 /().,%+-]{{1,60}}?)\s+"
    rf"(?P<value>{QUALITATIVE_VALUES})"
    rf"(?:\s+(?P<flag>{ABNORMAL_PATTERN}))?\s*$",
    re.I,
)

# Matches "Name ValueUnit [Range] [Flag]" — value is directly adjacent to unit (no space).
# Handles OCR output like "Glucose 103mg/dL 65-99 H".
ADJACENT_VALUE_UNIT_RE = re.compile(
    rf"^\s*(?P<name>[A-Za-z][A-Za-z0-9 /().,%+-]{{1,60}}?)\s+"
    rf"(?P<value>[<>]?\d+(?:\.\d+)?)(?P<unit>{UNIT_PATTERN})"
    rf"(?:\s+(?P<range>{RANGE_PATTERN}))?"
    rf"(?:\s+(?P<flag>{ABNORMAL_PATTERN}))?\s*$",
    re.I,
)

# Matches a line that is only a reference range (possibly with flag).
# Used to detect split-range rows where range appears on the line after the value.
_RANGE_ONLY_LINE_RE = re.compile(
    rf"^\s*(?P<range>{RANGE_PATTERN})\s*(?P<flag>{ABNORMAL_PATTERN})?\s*$",
    re.I,
)


@dataclass(frozen=True)
class LabRow:
    test_name: str
    value: str
    unit: str | None
    reference_range: str | None
    abnormal_flag: str | None
    raw_line: str
    parser_confidence: float
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_lab_rows(text: str) -> list[LabRow]:
    rows: list[LabRow] = []
    lines = candidate_lab_lines(text)
    skip_indices: set[int] = set()
    for i, line in enumerate(lines):
        if i in skip_indices:
            continue
        parsed = parse_lab_row(line)
        if parsed is not None:
            if parsed.reference_range is None and i + 1 < len(lines):
                upgraded, consumed = _try_attach_range_from_next_line(parsed, lines[i + 1])
                if upgraded is not None:
                    rows.append(upgraded)
                    if consumed:
                        skip_indices.add(i + 1)
                    continue
            rows.append(parsed)
    rows.extend(parse_paired_microbiology_rows(lines))
    return rows


def parse_lab_row(line: str) -> LabRow | None:
    cleaned = normalize_line(line)
    if not cleaned:
        return None
    warnings: list[str] = []
    # Try adjacent-value-unit first: it is more specific and would be falsely
    # consumed by NUMERIC_ROW_RE (which allows digits in the name group).
    match = ADJACENT_VALUE_UNIT_RE.match(cleaned)
    if match is not None:
        warnings.append("adjacent_value_unit")
    else:
        match = NUMERIC_ROW_RE.match(cleaned) or QUALITATIVE_ROW_RE.match(cleaned)
    if not match:
        return None
    name = clean_name(match.group("name"))
    value = str(match.group("value")).strip()
    if len(name) < 2 or _looks_like_sentence(name):
        return None
    unit = _clean_optional(match.groupdict().get("unit"))
    reference_range = _clean_optional(match.groupdict().get("range"))
    abnormal_flag = _clean_optional(match.groupdict().get("flag"))
    confidence = 0.72
    if unit:
        confidence += 0.08
    if reference_range:
        confidence += 0.10
    if abnormal_flag:
        confidence += 0.04
    if re.search(QUALITATIVE_VALUES, value, re.I):
        confidence += 0.06
    if not unit and not reference_range and not abnormal_flag:
        warnings.append("qualitative_or_sparse_row")
    return LabRow(
        test_name=name,
        value=value,
        unit=unit,
        reference_range=reference_range,
        abnormal_flag=abnormal_flag,
        raw_line=line.strip(),
        parser_confidence=round(min(confidence, 0.98), 3),
        warnings=warnings,
    )


def _try_attach_range_from_next_line(
    row: LabRow, next_line: str
) -> tuple[LabRow | None, bool]:
    """Attempt to upgrade a row missing a reference range by reading the next line.

    Returns (upgraded_row, consumed) where consumed=True means the next line
    was a range-only line and should be skipped in the main loop.
    """
    cleaned_next = normalize_line(next_line)
    if not cleaned_next:
        return None, False
    range_match = _RANGE_ONLY_LINE_RE.match(cleaned_next)
    if range_match is None:
        return None, False
    new_range = _clean_optional(range_match.group("range"))
    if new_range is None:
        return None, False
    new_flag = _clean_optional(range_match.groupdict().get("flag")) or row.abnormal_flag
    upgraded_warnings = [w for w in row.warnings if w != "qualitative_or_sparse_row"]
    upgraded_warnings.append("split_range_row")
    upgraded_confidence = round(min(row.parser_confidence + 0.10, 0.98), 3)
    return LabRow(
        test_name=row.test_name,
        value=row.value,
        unit=row.unit,
        reference_range=new_range,
        abnormal_flag=new_flag,
        raw_line=row.raw_line,
        parser_confidence=upgraded_confidence,
        warnings=upgraded_warnings,
    ), True


def parse_paired_microbiology_rows(lines: list[str]) -> list[LabRow]:
    rows: list[LabRow] = []
    organism_re = re.compile(r"\b(?:[A-Z][a-z]+(?:ella|bacter|coccus|plasma|dida|spp\.?)|Ureaplasma|Candida|Enterobacteriaceae|Enterococcus)\b", re.I)
    value_re = re.compile(r"\b(?P<value>10\s*\^?\s*\d(?:\.\d)?|10\s+\d(?:\.\d)?|not detected|none detected|detected)\b", re.I)
    for index, line in enumerate(lines):
        cleaned = normalize_line(line)
        if not organism_re.search(cleaned):
            continue
        window = " ".join(normalize_line(item) for item in lines[index + 1:index + 4])
        value_match = value_re.search(window)
        if value_match is None:
            value_match = value_re.search(cleaned)
        if value_match is None:
            continue
        value = re.sub(r"\s+", "^", value_match.group("value").strip(), count=1)
        rows.append(LabRow(
            test_name=clean_name(cleaned),
            value=value,
            unit="Lg" if value.lower().startswith("10") else None,
            reference_range=None,
            abnormal_flag=None,
            raw_line=line.strip(),
            parser_confidence=0.74,
            warnings=["paired_microbiology_row"],
        ))
    return rows


def candidate_lab_lines(text: str) -> list[str]:
    source = text or ""
    raw_lines: list[str] = []
    for line in source.splitlines():
        raw_lines.extend(split_flattened_line(line))
    return [line for line in raw_lines if len(line.strip()) >= 3]


def split_flattened_line(line: str) -> list[str]:
    cleaned = line.strip()
    if not cleaned:
        return []
    if "|" in cleaned:
        return [part.strip() for part in cleaned.split("|") if part.strip()]
    if "\t" in cleaned:
        return [part.strip() for part in cleaned.split("\t") if part.strip()]
    return [cleaned]


def normalize_line(line: str) -> str:
    cleaned = re.sub(r"\s+", " ", line).strip(" .:;")
    cleaned = re.sub(r"^(?:\[[^\]]+\]\s*)+", "", cleaned)
    return cleaned


def clean_name(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", name).strip(" .:-")
    aliases = {
        "rbc ua": "RBC UA",
        "wbc ua": "WBC UA",
    }
    return aliases.get(cleaned.lower(), cleaned)


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _looks_like_sentence(name: str) -> bool:
    words = name.split()
    return len(words) > 7


# ---------------------------------------------------------------------------
# Forensic / debug helpers (Phase 42)
#
# These functions explain WHY individual candidate lines were not parsed as
# lab rows. They do not change production parse_lab_rows behavior — they
# call the same parsers and only inspect the inputs that get rejected.
# ---------------------------------------------------------------------------

_NAME_ONLY_RE = re.compile(r"^[A-Za-z][A-Za-z /().,%+-]*$")
_VALUE_ONLY_RE = re.compile(r"^[\d<>.\s/-]+$")
_QUALITATIVE_TOKEN_RE = re.compile(QUALITATIVE_VALUES, re.I)
_UNIT_TOKEN_RE = re.compile(UNIT_PATTERN, re.I)


def _classify_rejection(line: str, cleaned: str) -> dict[str, Any]:
    """Categorize why a candidate line failed to parse as a lab row."""
    if not cleaned:
        return {
            "line": line,
            "reason": "empty_or_noise",
            "matched_partial_pattern": None,
            "notes": "empty after normalization",
        }

    if _RANGE_ONLY_LINE_RE.match(cleaned):
        return {
            "line": line,
            "reason": "range_only",
            "matched_partial_pattern": "range",
            "notes": "no analyte name attached on this line",
        }

    has_alpha = bool(re.search(r"[A-Za-z]{2,}", cleaned))
    has_digit = bool(re.search(r"\d", cleaned))
    has_unit_token = bool(_UNIT_TOKEN_RE.search(cleaned))
    has_qualitative = bool(_QUALITATIVE_TOKEN_RE.search(cleaned))

    if _NAME_ONLY_RE.match(cleaned) and not has_digit:
        if has_qualitative:
            return {
                "line": line,
                "reason": "qualitative_only",
                "matched_partial_pattern": "value",
                "notes": "qualitative term without analyte name on same line",
            }
        return {
            "line": line,
            "reason": "name_only",
            "matched_partial_pattern": "name",
            "notes": "alpha-only line, no value present",
        }

    if _VALUE_ONLY_RE.match(cleaned):
        return {
            "line": line,
            "reason": "value_only",
            "matched_partial_pattern": "value",
            "notes": "numeric-only line, no analyte name",
        }

    non_alnum = sum(1 for c in cleaned if not c.isalnum() and not c.isspace())
    noise_ratio = non_alnum / max(len(cleaned), 1)

    if has_alpha and has_digit:
        if not has_unit_token and not has_qualitative:
            return {
                "line": line,
                "reason": "no_unit_or_qualitative",
                "matched_partial_pattern": "name+value",
                "notes": "name and digits present but no recognized unit or qualitative term",
            }
        if noise_ratio > 0.30:
            return {
                "line": line,
                "reason": "noisy_high_punctuation",
                "matched_partial_pattern": None,
                "notes": f"non-alnum ratio {noise_ratio:.2f} above 0.30 threshold",
            }
        return {
            "line": line,
            "reason": "malformed_lab_row",
            "matched_partial_pattern": "partial",
            "notes": "name+value+unit-like content but failed all parser patterns",
        }

    if noise_ratio > 0.40:
        return {
            "line": line,
            "reason": "noisy_high_punctuation",
            "matched_partial_pattern": None,
            "notes": f"non-alnum ratio {noise_ratio:.2f} above 0.40 threshold",
        }

    return {
        "line": line,
        "reason": "unrecognized",
        "matched_partial_pattern": None,
        "notes": "no clear lab row structure",
    }


def _signal_summary(lines: list[str], rejected: list[dict[str, Any]]) -> dict[str, Any]:
    if not lines:
        return {
            "has_value_only_lines": False,
            "has_name_only_lines": False,
            "has_range_only_lines": False,
            "appears_table_like_no_separators": False,
            "text_too_fragmented": False,
            "text_too_sparse": True,
            "short_line_ratio": 0.0,
        }
    short_count = sum(1 for line in lines if len(line) < 10)
    short_ratio = short_count / len(lines)
    joined = "\n".join(lines)
    has_separators = "|" in joined or "\t" in joined
    table_like = sum(
        1 for line in lines
        if re.search(r"[A-Za-z]{2,}\s+\d", line) and re.search(r"\d\s+[A-Za-z]", line)
    )
    return {
        "has_value_only_lines": any(r["reason"] == "value_only" for r in rejected),
        "has_name_only_lines": any(r["reason"] == "name_only" for r in rejected),
        "has_range_only_lines": any(r["reason"] == "range_only" for r in rejected),
        "appears_table_like_no_separators": table_like >= 2 and not has_separators,
        "text_too_fragmented": short_ratio > 0.5,
        "text_too_sparse": len(lines) < 3,
        "short_line_ratio": round(short_ratio, 3),
    }


def debug_parse_lab_lines(text: str) -> dict[str, Any]:
    """Forensic helper: report why each candidate line was/was not parsed.

    Does NOT alter production parse_lab_rows behavior — it calls the same
    parsers and only categorizes the inputs that get rejected.
    """
    lines = candidate_lab_lines(text)
    parsed_entries: list[dict[str, Any]] = []
    rejected_entries: list[dict[str, Any]] = []

    for index, line in enumerate(lines):
        row = parse_lab_row(line)
        if row is not None:
            parsed_entries.append({"index": index, "line": line, "row": row.to_dict()})
            continue
        cleaned = normalize_line(line)
        info = _classify_rejection(line, cleaned)
        info["index"] = index
        rejected_entries.append(info)

    breakdown: dict[str, int] = {}
    for entry in rejected_entries:
        breakdown[entry["reason"]] = breakdown.get(entry["reason"], 0) + 1

    return {
        "candidate_line_count": len(lines),
        "parsed_row_count": len(parsed_entries),
        "rejected_line_count": len(rejected_entries),
        "candidate_lines": list(lines[:50]),
        "parsed_rows": parsed_entries[:50],
        "rejected_lines": rejected_entries[:50],
        "rejection_reason_breakdown": breakdown,
        "signal_summary": _signal_summary(lines, rejected_entries),
    }
