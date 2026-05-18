"""Numeric-table safe-default language-visibility metadata label.

This module implements MEDAI-DOC-TYPE-UNKNOWN-DIAG-06A-IMPLEMENTATION.

It exposes a single pure function:

    derive_numeric_table_safe_default_label(record, *, enabled=False) -> str | None

The function returns the controlled-vocabulary label

    ``"latin_script_likely_english_table_context"``

only when every condition below is true:

    1. The caller explicitly passes ``enabled=True``. The function is
       default-off; the disable / rollback path is simply to omit the
       parameter or pass ``enabled=False``.
    2. The record matches all 14 fields of the positive signature.
    3. None of the 10 exclusion rules fires.

Otherwise the function returns ``None``.

What this module DOES NOT do
----------------------------
* Does NOT modify the raw language-detector output. The derived label is a
  separate piece of safe metadata; the caller decides whether to attach it
  to the record.
* Does NOT auto-accept anything. The label is for routing / metadata only.
* Does NOT classify clinical meaning.
* Does NOT parse lab values, medications, doses, frequencies, durations,
  or DDIs.
* Does NOT write active clinical facts.
* Does NOT change OCR routing, OCR engine, classifier behavior, thresholds,
  scoring, cue packs, B07, ROUTE-FIX, DB schema, command allowlist, or
  external API behavior.
* Does NOT change review-bound status: callers must continue to treat any
  document carrying this derived label as review-bound.

The function is intentionally additive: callers must opt in. Existing
pipelines that do not import this module are unaffected.
"""
from __future__ import annotations

from typing import Any, Mapping

# ── Controlled-vocabulary constants ──────────────────────────────────────────

DERIVED_LABEL = "latin_script_likely_english_table_context"

# 14-field positive signature. The first element is a synthetic label name
# used in spec documents; the second is the per-record predicate. Returning
# False from any predicate disables the derived label.
POSITIVE_SIGNATURE: tuple[tuple[str, str], ...] = (
    ("table_like_structure_detected",                "yes"),
    ("high_table_density_required",                  "yes"),
    ("repeated_row_pattern_visible",                 "yes"),
    ("numeric_content_bucket",                       "medium"),
    ("numeric_units_or_ranges_visible",              "yes"),
    ("script_detection_result",                      "latin"),
    ("dominant_script",                              "latin"),
    ("detector_confidence_bucket",                   "high"),
    ("alphabetic_ratio_sufficient_for_language",     "yes"),
    ("administrative_table_shape",                   "yes"),
    ("treatment_schedule_table_shape",               "yes"),
    ("language_detector_attempted",                  "yes"),
    ("language_detector_input_bucket",               "sufficient"),
    ("language_visibility_status",                   "latin_visible_language_unknown"),
)

EXCLUSION_RULES: tuple[str, ...] = (
    "exclude_cyrillic_dominant_records",
    "exclude_mixed_script_records",
    "exclude_low_alphabetic_ratio_records",
    "exclude_no_text_layer_records",
    "exclude_image_like_but_not_routed_records",
    "exclude_ambiguous_below_threshold_records",
    "exclude_fallback_ran_but_no_family_match_records",
    "exclude_medication_dose_or_ddi_interpretation",
    "exclude_lab_value_parsing",
    "exclude_records_with_insufficient_safe_metadata",
)


# ── Internal helpers ─────────────────────────────────────────────────────────

def _g(record: Mapping[str, Any], key: str) -> str:
    """Lowercase-string fetch with safe defaults."""
    value = record.get(key)
    if value is None:
        return ""
    return str(value).strip().lower()


def _positive_signature_predicate(
    key: str,
    expected: str,
    record: Mapping[str, Any],
) -> bool:
    """Return True if ``record`` satisfies signature field ``key``.

    The mapping from synthetic spec keys to raw record fields is identical to
    the one validated in DIAG-06A's spec block.
    """
    if key == "table_like_structure_detected":
        return _g(record, "table_like_structure_detected") == expected
    if key == "high_table_density_required":
        return _g(record, "table_like_structure_detected") == "yes"
    if key == "repeated_row_pattern_visible":
        return (
            _g(record, "table_like_structure_detected") == "yes"
            and _g(record, "date_or_schedule_shape_detected") == "yes"
        )
    if key == "numeric_content_bucket":
        return _g(record, "numeric_content_bucket") == expected
    if key == "numeric_units_or_ranges_visible":
        return _g(record, "symbol_content_bucket") in {"medium", "high"}
    if key == "script_detection_result":
        return _g(record, "script_detection_result") == expected
    if key == "dominant_script":
        return _g(record, "dominant_script") == expected
    if key == "detector_confidence_bucket":
        return _g(record, "detector_confidence_bucket") == expected
    if key == "alphabetic_ratio_sufficient_for_language":
        return _g(record, "alphabetic_content_bucket") == "high"
    if key == "administrative_table_shape":
        return _g(record, "administrative_form_shape_detected") == "yes"
    if key == "treatment_schedule_table_shape":
        return _g(record, "date_or_schedule_shape_detected") == "yes"
    if key == "language_detector_attempted":
        return _g(record, "language_detector_attempted") == "yes"
    if key == "language_detector_input_bucket":
        return _g(record, "language_detector_input_bucket") == expected
    if key == "language_visibility_status":
        return _g(record, "language_visibility_status") == expected
    return False


def matches_positive_signature(record: Mapping[str, Any]) -> bool:
    """Return True iff every field of the 14-field positive signature holds."""
    return all(
        _positive_signature_predicate(key, expected, record)
        for key, expected in POSITIVE_SIGNATURE
    )


def _exclusion_rule_fires(rule: str, record: Mapping[str, Any]) -> bool:
    """Return True iff the given exclusion rule fires on the record."""
    if rule == "exclude_cyrillic_dominant_records":
        return _g(record, "dominant_script") == "cyrillic"
    if rule == "exclude_mixed_script_records":
        return _g(record, "dominant_script") == "mixed"
    if rule == "exclude_low_alphabetic_ratio_records":
        return _g(record, "alphabetic_content_bucket") == "low"
    if rule == "exclude_no_text_layer_records":
        return _g(record, "pdf_text_layer_detected") == "no"
    if rule == "exclude_image_like_but_not_routed_records":
        return _g(record, "image_like_pdf") == "yes"
    if rule == "exclude_ambiguous_below_threshold_records":
        return _g(record, "unknown_failure_bucket") == "ambiguous_below_threshold"
    if rule == "exclude_fallback_ran_but_no_family_match_records":
        return (
            _g(record, "unknown_failure_bucket") == "fallback_ran_but_no_family_match"
        )
    if rule == "exclude_medication_dose_or_ddi_interpretation":
        # Structural compliance: callers must not pass parsed medication
        # / dose / DDI fields through this helper. The privacy-safe per-file
        # table never carries those, so the rule defends a contract rather
        # than a present-day record. We treat any presence of these caller-
        # provided fields as a violation.
        for forbidden in (
            "parsed_medications",
            "parsed_doses",
            "parsed_frequencies",
            "parsed_ddi_findings",
            "ddi_interpretation",
        ):
            if record.get(forbidden):
                return True
        return False
    if rule == "exclude_lab_value_parsing":
        for forbidden in ("parsed_lab_values", "lab_value_interpretation"):
            if record.get(forbidden):
                return True
        return False
    if rule == "exclude_records_with_insufficient_safe_metadata":
        required = (
            "table_like_structure_detected",
            "numeric_content_bucket",
            "alphabetic_content_bucket",
            "dominant_script",
            "detector_confidence_bucket",
            "language_visibility_status",
        )
        return any(_g(record, k) == "" for k in required)
    # Unknown rule -> defensively fire (treat as a violation).
    return True


def violates_any_exclusion_rule(record: Mapping[str, Any]) -> bool:
    """Return True iff any of the 10 exclusion rules fires on the record."""
    return any(_exclusion_rule_fires(rule, record) for rule in EXCLUSION_RULES)


# ── Implementation-level safeguards (beyond the 14-field signature) ──────────
#
# The DIAG-06A spec's 14-field positive signature was validated on the 11
# priority records. On the full 507-row corpus, the same 14 fields also match
# a handful of additional records that DIAG-02 / DIAG-03 / DIAG-04 / DIAG-05
# routed to OTHER levers (table-header / abbreviation / text-layer). To
# satisfy the DIAG-06A acceptance criterion
#
#     unknown_count_decreases_only_for_exact_signature_records
#
# the helper applies a small set of implementation-level safeguards that
# enforce the upstream-filter conditions implicit in the priority slice.
# These safeguards never make the helper LESS conservative; they only narrow
# the set of records that can receive the derived label.


_IMPLEMENTATION_SAFEGUARDS: tuple[str, ...] = (
    "must_be_predicted_document_type_unknown",
    "must_be_in_insufficient_text_visibility_bucket",
    "must_not_be_in_text_layer_present_but_too_short_routing",
    "must_have_no_section_heading_shape_detected",
    "must_have_no_medical_abbreviation_shape_detected",
)


def _safeguard_fires(rule: str, record: Mapping[str, Any]) -> bool:
    if rule == "must_be_predicted_document_type_unknown":
        return _g(record, "predicted_document_type") != "unknown"
    if rule == "must_be_in_insufficient_text_visibility_bucket":
        return _g(record, "unknown_failure_bucket") != "insufficient_text_visibility"
    if rule == "must_not_be_in_text_layer_present_but_too_short_routing":
        return (
            _g(record, "unknown_ocr_routing_bucket")
            == "text_layer_present_but_too_short"
        )
    if rule == "must_have_no_section_heading_shape_detected":
        return _g(record, "section_heading_shape_detected") == "yes"
    if rule == "must_have_no_medical_abbreviation_shape_detected":
        return _g(record, "medical_abbreviation_shape_detected") == "yes"
    return True


def _fails_any_implementation_safeguard(record: Mapping[str, Any]) -> bool:
    return any(_safeguard_fires(g, record) for g in _IMPLEMENTATION_SAFEGUARDS)


# ── Public API ───────────────────────────────────────────────────────────────

def derive_numeric_table_safe_default_label(
    record: Mapping[str, Any],
    *,
    enabled: bool = False,
) -> str | None:
    """Return the safe metadata label, or ``None``.

    :param record:  per-file record-shaped mapping (the same shape used in
                    the FAMILY-04 anonymized per-file table). Must NOT carry
                    parsed lab values, medications, doses, frequencies, or
                    DDI findings.
    :param enabled: explicit opt-in. Default-off. Pass ``enabled=False`` (or
                    omit) to disable the derivation; this is the rollback
                    path.
    :returns:       ``"latin_script_likely_english_table_context"`` when the
                    helper is enabled, every positive-signature field holds,
                    and no exclusion rule fires; otherwise ``None``.

    The function is pure. It never mutates ``record``. It never modifies
    raw detector output. It never auto-accepts, classifies clinical
    meaning, parses values, or writes active clinical facts. The caller
    must continue to treat any record carrying this derived label as
    review-bound.
    """
    if not enabled:
        return None
    if not matches_positive_signature(record):
        return None
    if violates_any_exclusion_rule(record):
        return None
    if _fails_any_implementation_safeguard(record):
        return None
    return DERIVED_LABEL


def is_disabled_by_default() -> bool:
    """Convenience predicate for callers that want to inspect the default."""
    return True
