"""Deterministic lab table normalization and coverage recovery."""

from lab_normalization.lab_coverage import LabCoverage, measure_lab_coverage
from lab_normalization.lab_normalizer import LabNormalizationResult, normalize_lab_text
from lab_normalization.lab_row_parser import LabRow, parse_lab_rows
from lab_normalization.lab_table_detector import LabTableDetection, detect_lab_table

__all__ = [
    "LabCoverage",
    "LabNormalizationResult",
    "LabRow",
    "LabTableDetection",
    "detect_lab_table",
    "measure_lab_coverage",
    "normalize_lab_text",
    "parse_lab_rows",
]
