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
    for line in lines:
        parsed = parse_lab_row(line)
        if parsed is not None:
            rows.append(parsed)
    rows.extend(parse_paired_microbiology_rows(lines))
    return rows


def parse_lab_row(line: str) -> LabRow | None:
    cleaned = normalize_line(line)
    if not cleaned:
        return None
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
    warnings: list[str] = []
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
    return [line for line in raw_lines if len(line.strip()) >= 5]


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
