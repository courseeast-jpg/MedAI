from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


LAB_SECTION_PATTERNS = {
    "cbc": re.compile(r"\b(?:cbc|complete blood count|wbc|rbc|hemoglobin|hematocrit|platelets?)\b", re.I),
    "cmp": re.compile(r"\b(?:cmp|comprehensive metabolic panel|metabolic panel|creatinine|bilirubin|albumin|sodium|potassium)\b", re.I),
    "urinalysis": re.compile(r"\b(?:urinalysis|urine|specific gravity|ketones|nitrite|leukocytes?|occult blood|rbc ua|wbc ua)\b", re.I),
    "lipid_panel": re.compile(r"\b(?:lipid panel|cholesterol|triglycerides|hdl|ldl)\b", re.I),
    "blood_test": re.compile(r"\b(?:blood test|test results|lab(?:oratory)? results?|specimen)\b", re.I),
    "microbiology_pcr": re.compile(r"\b(?:pcr|microflora|androflor|candida|ureaplasma|enterobacteriaceae|enterococcus|spp\.)\b", re.I),
}

LAB_ROW_SIGNAL_RE = re.compile(
    r"\b[A-Za-z][A-Za-z /().%-]{1,45}\s+"
    r"(?:[<>]?\d+(?:\.\d+)?|negative|positive|trace|none detected|not detected)\b"
    r"(?:\s*[A-Za-z/%0-9^.-]+)?"
    r"(?:\s+\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?)?"
    r"(?:\s+(?:H|L|High|Low|Abnormal))?\b",
    re.I,
)


@dataclass(frozen=True)
class LabTableDetection:
    has_lab_table: bool
    detected_sections: list[str]
    table_confidence: float
    warnings: list[str]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_lab_table(text: str) -> LabTableDetection:
    source = text or ""
    lines = [line.strip() for line in source.splitlines() if line.strip()]
    detected_sections = [
        name for name, pattern in LAB_SECTION_PATTERNS.items() if pattern.search(source)
    ]
    row_signal_count = sum(1 for line in lines if LAB_ROW_SIGNAL_RE.search(line))
    numeric_unit_count = sum(1 for line in lines if re.search(r"\d+(?:\.\d+)?\s*(?:mg/dL|x10E3/uL|/hpf|mmol/L|CFU/mL)\b", line, re.I))
    qualitative_count = sum(1 for line in lines if re.search(r"\b(?:negative|positive|trace|not detected|none detected)\b", line, re.I))
    microbiology_count = sum(1 for line in lines if re.search(r"\b(?:candida|ureaplasma|enterobacteriaceae|enterococcus|spp\.|10\s+\d(?:\.\d)?)\b", line, re.I))
    table_marker_count = sum(source.count(marker) for marker in ("|", "\t", "  ", "....", "----"))

    signal_score = min(row_signal_count / 4.0, 1.0)
    section_score = min(len(detected_sections) / 2.0, 1.0)
    value_score = min((numeric_unit_count + qualitative_count + microbiology_count) / 5.0, 1.0)
    table_confidence = round((0.45 * signal_score) + (0.30 * section_score) + (0.25 * value_score), 3)
    has_lab_table = table_confidence >= 0.45 or (row_signal_count >= 2 and bool(detected_sections))

    warnings: list[str] = []
    if not lines:
        warnings.append("empty_text")
    if table_marker_count >= 4:
        warnings.append("flattened_table_markers")
    if row_signal_count == 0 and detected_sections:
        warnings.append("lab_section_without_rows")

    return LabTableDetection(
        has_lab_table=has_lab_table,
        detected_sections=detected_sections,
        table_confidence=table_confidence,
        warnings=warnings,
        metadata={
            "line_count": len(lines),
            "row_signal_count": row_signal_count,
            "numeric_unit_count": numeric_unit_count,
            "qualitative_count": qualitative_count,
            "microbiology_count": microbiology_count,
            "table_marker_count": table_marker_count,
        },
    )
