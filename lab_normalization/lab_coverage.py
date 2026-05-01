from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from lab_normalization.lab_row_parser import LabRow, candidate_lab_lines


@dataclass(frozen=True)
class LabCoverage:
    raw_candidate_line_count: int
    parsed_lab_row_count: int
    coverage_ratio: float
    coverage_band: str
    reason_codes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def measure_lab_coverage(text: str, parsed_rows: list[LabRow]) -> LabCoverage:
    candidate_count = len(candidate_lab_lines(text))
    parsed_count = len(parsed_rows)
    ratio = round(parsed_count / max(candidate_count, 1), 3)
    band = coverage_band(parsed_count=parsed_count, ratio=ratio)
    reasons = reason_codes_for(parsed_count=parsed_count, ratio=ratio, band=band)
    return LabCoverage(
        raw_candidate_line_count=candidate_count,
        parsed_lab_row_count=parsed_count,
        coverage_ratio=ratio,
        coverage_band=band,
        reason_codes=reasons,
    )


def coverage_band(*, parsed_count: int, ratio: float) -> str:
    if parsed_count == 0:
        return "none"
    if parsed_count >= 6 and ratio >= 0.18:
        return "good"
    if parsed_count >= 3 and ratio >= 0.08:
        return "partial"
    return "weak"


def reason_codes_for(*, parsed_count: int, ratio: float, band: str) -> list[str]:
    if band == "none":
        return ["lab_table_not_recovered"]
    reasons = ["lab_table_recovered"]
    if band == "partial":
        reasons.append("lab_table_partial_coverage")
    if band == "weak":
        reasons.append("lab_table_weak_coverage")
    if ratio < 0.10:
        reasons.append("lab_table_low_ratio")
    if parsed_count >= 3:
        reasons.append("lab_rows_minimum_met")
    return reasons
