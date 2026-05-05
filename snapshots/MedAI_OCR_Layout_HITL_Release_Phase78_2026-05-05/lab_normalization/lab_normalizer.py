from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from document_classification import classify_document
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
    document_classification: dict[str, Any] | None = None
    skipped_for_document_type: bool = False

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
    classification = classify_document(text or "")
    classification_payload = classification.to_dict()

    if not classification.should_run_lab_normalization:
        return _skipped_result(
            classification=classification,
            classification_payload=classification_payload,
        )

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
        "document_type_eligible_for_lab_recovery": classification.document_type
        in {"lab_report", "unknown_medical", "microbiology_pcr_report"},
    }
    should_recover = all(safety_checks.values())
    reason_codes = list(coverage.reason_codes)
    if detection.has_lab_table:
        reason_codes.append("lab_table_detected")
    if classification.document_type == "lab_report" and classification.confidence >= 0.55:
        reason_codes.append("lab_report_detected")
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
        document_classification=classification_payload,
        skipped_for_document_type=False,
    )


def _skipped_result(
    *,
    classification: Any,
    classification_payload: dict[str, Any],
) -> LabNormalizationResult:
    """Produce a skipped LabNormalizationResult when classifier blocks parsing."""
    reason_codes: list[str] = ["non_lab_document_skipped_lab_normalization"]
    if classification.document_type == "prescription":
        reason_codes.append("document_type_prescription_not_lab")
    elif classification.document_type == "microbiology_pcr_report":
        reason_codes.append("microbiology_pcr_report_detected")
    elif classification.document_type == "imaging_report":
        reason_codes.append("imaging_report_detected")
    if classification.should_recommend_language_aware_ocr:
        reason_codes.append("language_aware_ocr_required")
    if classification.confidence < 0.50:
        reason_codes.append("low_confidence_document_type")

    safety_checks = {
        "ocr_layout_quality_safe": False,
        "lab_table_detected": False,
        "parsed_lab_row_minimum_met": False,
        "coverage_recovered": False,
        "no_empty_extraction_leakage": False,
        "not_poor_ocr_band": True,
        "no_safety_gate_block": True,
        "current_status_review_ocr_quality": False,
        "document_type_eligible_for_lab_recovery": False,
    }

    return LabNormalizationResult(
        normalized_lab_rows=[],
        lab_table_detected=False,
        lab_coverage_ratio=0.0,
        lab_coverage_band="none",
        lab_reason_codes=dedupe(reason_codes),
        should_upgrade_from_ocr_review_to_review=False,
        detection={
            "skipped_for_document_type": classification.document_type,
            "language_hint": classification.language_hint,
        },
        coverage={
            "raw_candidate_line_count": 0,
            "parsed_lab_row_count": 0,
            "coverage_ratio": 0.0,
            "coverage_band": "none",
            "reason_codes": list(reason_codes),
        },
        safety_checks=safety_checks,
        document_classification=classification_payload,
        skipped_for_document_type=True,
    )


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
