from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from lab_normalization.lab_coverage import measure_lab_coverage
from lab_normalization.lab_row_parser import LabRow, parse_lab_rows
from lab_normalization.lab_table_detector import detect_lab_table


SAFE_MIN_PARSED_ROWS = 3


@dataclass(frozen=True)
class LabNormalizationResult:
    normalized_lab_rows: list[LabRow]
    lab_table_detected: bool
    lab_coverage_ratio: float
    lab_coverage_band: str
    lab_reason_codes: list[str]
    should_upgrade_from_ocr_review_to_review: bool
    detection: dict[str, Any]
    coverage: dict[str, Any]
    safety_checks: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["normalized_lab_rows"] = [row.to_dict() for row in self.normalized_lab_rows]
        return payload


def normalize_lab_text(
    text: str,
    *,
    ocr_layout_quality_band: str | None,
    current_status: str,
    entity_count: int,
    safety_gate_blocked: bool = False,
) -> LabNormalizationResult:
    detection = detect_lab_table(text)
    rows = parse_lab_rows(text)
    coverage = measure_lab_coverage(text, rows)
    quality_band = str(ocr_layout_quality_band or "unknown")
    safety_checks = {
        "ocr_layout_quality_safe": quality_band in {"good", "usable_with_review"},
        "lab_table_detected": detection.has_lab_table,
        "parsed_lab_row_minimum_met": coverage.parsed_lab_row_count >= SAFE_MIN_PARSED_ROWS,
        "coverage_recovered": coverage.coverage_band in {"good", "partial"},
        "no_empty_extraction_leakage": entity_count > 0,
        "not_poor_ocr_band": quality_band not in {"poor_ocr", "empty"},
        "no_safety_gate_block": not safety_gate_blocked,
        "current_status_review_ocr_quality": current_status == "review_ocr_quality",
    }
    should_recover = all(safety_checks.values())
    reason_codes = list(coverage.reason_codes)
    if detection.has_lab_table:
        reason_codes.append("lab_table_detected")
    if should_recover:
        reason_codes.append("lab_table_recovered_review_only")
    return LabNormalizationResult(
        normalized_lab_rows=rows,
        lab_table_detected=detection.has_lab_table,
        lab_coverage_ratio=coverage.coverage_ratio,
        lab_coverage_band=coverage.coverage_band,
        lab_reason_codes=dedupe(reason_codes),
        should_upgrade_from_ocr_review_to_review=should_recover,
        detection=detection.to_dict(),
        coverage=coverage.to_dict(),
        safety_checks=safety_checks,
    )


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
