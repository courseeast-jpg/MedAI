"""Report count and status taxonomy invariants for validation summaries."""

from __future__ import annotations

from typing import Any


STATUS_TAXONOMY = {"accepted", "review", "review_ocr_quality", "empty", "error"}
COUNT_CONVENTION = "overlapping_review_total_with_review_ocr_quality_and_empty_subsets"
COUNT_CONVENTION_EXPLANATION = (
    "accepted and review are mutually exclusive top-level buckets; review includes "
    "review_ocr_quality, while empty is an extraction flag subset that can overlap "
    "with review. Therefore total == accepted + review, review_ocr_quality <= review, "
    "and empty <= review."
)


def count_consistency_for_counts(counts: dict[str, int]) -> dict[str, Any]:
    total = int(counts.get("total_files") or counts.get("total") or 0)
    accepted = int(counts.get("accepted") or 0)
    review = int(counts.get("review") or 0)
    review_ocr_quality = int(counts.get("review_ocr_quality") or 0)
    empty = int(counts.get("empty") or 0)
    checks = {
        "accepted_plus_review_equals_total": accepted + review == total,
        "review_ocr_quality_is_review_subset": review_ocr_quality <= review,
        "empty_is_review_subset_or_zero": empty <= review,
    }
    return {
        "count_convention": COUNT_CONVENTION,
        "count_consistency_passed": all(checks.values()),
        "checks": checks,
        "explanation": COUNT_CONVENTION_EXPLANATION,
    }


def unexpected_statuses(statuses: list[str]) -> list[str]:
    return sorted({status for status in statuses if status not in STATUS_TAXONOMY})
