"""MEDAI-DOC-TYPE-UNKNOWN-DIAG-11A-IMPLEMENTATION - Latin medical abbreviation
metadata helper.

Implements the spec published in UNKNOWN-DIAG-11A (commit 7f248ff). Exposes:

    derive_latin_medical_abbreviation_metadata_label(record, *, enabled=False, env=None)

Returns ``"latin_medical_abbreviation_context"`` only when ALL of:

    1. The helper is explicitly enabled via ``enabled=True`` OR the SEPARATE
       env var ``MEDAI_DOC_TYPE_LATIN_ABBREVIATION_METADATA_ENABLED`` is
       truthy. The env var is intentionally DISTINCT from:
           * ``MEDAI_DOC_TYPE_OPERATOR_REVIEW_BADGE_ENABLED`` (DIAG-07A/08A)
           * ``MEDAI_DOC_TYPE_LANGUAGE_PROPAGATION_METADATA_ENABLED`` (DIAG-09A/10A)
    2. Every field of the 14-field positive signal pattern holds.
    3. None of the 14 exclusion rules fires.
    4. The record does NOT overlap with the DIAG-06A numeric-table safe-
       default pool (helper consulted directly).
    5. The record does NOT overlap with the DIAG-09A language-propagation
       pool (helper consulted directly).

The label means only: Latin-script text contains medical-style
abbreviations useful for language / context routing.

What this module does NOT do
----------------------------
* Does NOT parse the abbreviation.
* Does NOT expand the abbreviation.
* Does NOT classify clinical meaning.
* Does NOT parse lab values / medications / doses / frequencies / DDIs.
* Does NOT auto-accept.
* Does NOT change data-layer document type.
* Does NOT modify raw language-detector output.
* Does NOT write active clinical facts.
* Does NOT change OCR routing, OCR engine, classifier behavior,
  thresholds, scoring, cue packs, B07, ROUTE-FIX, DB schema, command
  allowlist, or external API behavior.

Default-off
-----------
With no kwarg and the SEPARATE env var unset (or falsy), the function
always returns ``None``. Existing pipelines that never import the module
are unaffected.

Rollback paths
--------------
Any one of these is sufficient to disable:
    * Omit the ``enabled`` kwarg AND leave the env var unset.
    * Set the env var to a falsy value (0 / false / no / off / disabled).
    * Pass ``enabled=False`` explicitly (overrides truthy env).
    * Never import the module.
"""
from __future__ import annotations

import os
from typing import Any, Mapping, Optional

from clinical_knowledge.document_type.language_propagation_metadata import (
    derive_language_propagation_metadata_label,
)
from clinical_knowledge.document_type.numeric_table_safe_default import (
    derive_numeric_table_safe_default_label,
)

# ── Controlled-vocabulary constants ──────────────────────────────────────────

PROPAGATED_METADATA_LABEL = "latin_medical_abbreviation_context"
LATIN_ABBREVIATION_METADATA_LABEL = PROPAGATED_METADATA_LABEL  # alias

PROPAGATED_METADATA_DISCLAIMER = (
    "Safe metadata only. Indicates Latin-script medical-style abbreviations "
    "for language and context routing. The abbreviation is not parsed and "
    "not expanded. Not a final document type. Not clinical interpretation. "
    "Raw detector output unchanged."
)
LATIN_ABBREVIATION_METADATA_DISCLAIMER = PROPAGATED_METADATA_DISCLAIMER  # alias

# SEPARATE env var. Deliberately distinct from the DIAG-07A and DIAG-09A
# levers so rolling forward / rolling back any one of the three is
# independent.
LATIN_ABBREVIATION_METADATA_ENV_VAR = (
    "MEDAI_DOC_TYPE_LATIN_ABBREVIATION_METADATA_ENABLED"
)

POSITIVE_SIGNAL_PATTERN: tuple[tuple[str, str], ...] = (
    ("detector_attempted",                       "yes"),
    ("detector_input_bucket",                    "sufficient"),
    ("detector_confidence_bucket",               "high_or_medium"),
    ("script_detection_result",                  "latin"),
    ("dominant_script",                          "latin"),
    ("language_visibility_status",               "latin_visible_language_unknown"),
    ("latin_medical_abbrev_visible",             "yes"),
    ("medical_abbreviation_shape_detected",      "yes"),
    ("alphabetic_ratio_sufficient_for_language", "yes"),
    ("no_cyrillic_dominant_signal",              "yes"),
    ("no_mixed_script_signal",                   "yes"),
    ("no_low_confidence_detector_signal",        "yes"),
    ("not_already_handled_by_numeric_table_safe_default_helper", "yes"),
    ("not_already_handled_by_language_propagation_helper",       "yes"),
)

EXCLUSION_RULES: tuple[str, ...] = (
    "exclude_cyrillic_dominant_records",
    "exclude_mixed_script_records",
    "exclude_low_detector_confidence_records",
    "exclude_insufficient_detector_input_records",
    "exclude_no_text_layer_records",
    "exclude_image_like_but_not_routed_records",
    "exclude_table_heavy_numeric_safe_default_records_already_handled",
    "exclude_language_propagation_records_already_handled",
    "exclude_table_header_only_special_case_record",
    "exclude_ambiguous_below_threshold_records",
    "exclude_fallback_ran_but_no_family_match_records",
    "exclude_medication_dose_or_ddi_interpretation",
    "exclude_lab_value_parsing",
    "exclude_records_with_insufficient_safe_metadata",
)

_TRUTHY = frozenset({"1", "true", "yes", "on", "enabled"})


# ── Internal helpers ─────────────────────────────────────────────────────────

def _g(record: Mapping[str, Any], key: str) -> str:
    v = record.get(key)
    if v is None:
        return ""
    return str(v).strip().lower()


def _positive_signal_predicate(
    key: str, expected: str, record: Mapping[str, Any],
) -> bool:
    if key == "detector_attempted":
        return _g(record, "language_detector_attempted") == expected
    if key == "detector_input_bucket":
        return _g(record, "language_detector_input_bucket") == expected
    if key == "detector_confidence_bucket":
        return _g(record, "detector_confidence_bucket") in {"high", "medium"}
    if key == "script_detection_result":
        return _g(record, "script_detection_result") == expected
    if key == "dominant_script":
        return _g(record, "dominant_script") == expected
    if key == "language_visibility_status":
        return _g(record, "language_visibility_status") == expected
    if key == "latin_medical_abbrev_visible":
        return (
            _g(record, "medical_abbreviation_shape_detected") == "yes"
            and _g(record, "dominant_script") == "latin"
        )
    if key == "medical_abbreviation_shape_detected":
        return _g(record, "medical_abbreviation_shape_detected") == expected
    if key == "alphabetic_ratio_sufficient_for_language":
        return _g(record, "alphabetic_content_bucket") == "high"
    if key == "no_cyrillic_dominant_signal":
        return _g(record, "dominant_script") != "cyrillic"
    if key == "no_mixed_script_signal":
        return _g(record, "dominant_script") != "mixed"
    if key == "no_low_confidence_detector_signal":
        return _g(record, "detector_confidence_bucket") != "low"
    if key == "not_already_handled_by_numeric_table_safe_default_helper":
        return derive_numeric_table_safe_default_label(record, enabled=True) is None
    if key == "not_already_handled_by_language_propagation_helper":
        return derive_language_propagation_metadata_label(record, enabled=True) is None
    return False


def matches_positive_signal_pattern(record: Mapping[str, Any]) -> bool:
    """Return True iff every field of the 14-field positive signal pattern holds."""
    return all(
        _positive_signal_predicate(k, v, record)
        for k, v in POSITIVE_SIGNAL_PATTERN
    )


def _exclusion_rule_fires(rule: str, record: Mapping[str, Any]) -> bool:
    if rule == "exclude_cyrillic_dominant_records":
        return _g(record, "dominant_script") == "cyrillic"
    if rule == "exclude_mixed_script_records":
        return _g(record, "dominant_script") == "mixed"
    if rule == "exclude_low_detector_confidence_records":
        return _g(record, "detector_confidence_bucket") == "low"
    if rule == "exclude_insufficient_detector_input_records":
        return _g(record, "language_detector_input_bucket") in {
            "insufficient", "absent",
        }
    if rule == "exclude_no_text_layer_records":
        return _g(record, "pdf_text_layer_detected") == "no"
    if rule == "exclude_image_like_but_not_routed_records":
        return _g(record, "image_like_pdf") == "yes"
    if rule == "exclude_table_heavy_numeric_safe_default_records_already_handled":
        return (
            derive_numeric_table_safe_default_label(record, enabled=True)
            is not None
        )
    if rule == "exclude_language_propagation_records_already_handled":
        return (
            derive_language_propagation_metadata_label(record, enabled=True)
            is not None
        )
    if rule == "exclude_table_header_only_special_case_record":
        # A record routed to the table-header special case has
        # section_heading=yes AND no abbreviation signal.
        return (
            _g(record, "section_heading_shape_detected") == "yes"
            and _g(record, "medical_abbreviation_shape_detected") != "yes"
        )
    if rule == "exclude_ambiguous_below_threshold_records":
        return (
            _g(record, "unknown_failure_bucket")
            == "ambiguous_below_threshold"
        )
    if rule == "exclude_fallback_ran_but_no_family_match_records":
        return (
            _g(record, "unknown_failure_bucket")
            == "fallback_ran_but_no_family_match"
        )
    if rule == "exclude_medication_dose_or_ddi_interpretation":
        for forbidden in (
            "parsed_medications", "parsed_doses", "parsed_frequencies",
            "parsed_ddi_findings", "ddi_interpretation",
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
            "language_detector_attempted",
            "language_detector_input_bucket",
            "detector_confidence_bucket",
            "script_detection_result",
            "dominant_script",
            "language_visibility_status",
            "alphabetic_content_bucket",
            "medical_abbreviation_shape_detected",
        )
        return any(_g(record, k) == "" for k in required)
    return True


def violates_any_exclusion_rule(record: Mapping[str, Any]) -> bool:
    """Return True iff any of the 14 exclusion rules fires on the record."""
    return any(_exclusion_rule_fires(rule, record) for rule in EXCLUSION_RULES)


# ── Implementation-level safeguards (beyond the 14-field signature) ──────────
#
# The DIAG-11A spec's 14-field positive signal pattern was validated on the 8
# priority records. On the full 507-row corpus the same 14 fields can match
# additional records DIAG-02 / DIAG-03 / DIAG-04 route to other levers OR
# records that are not even Unknown. The safeguards close that gap so the
# helper labels only the exact 8 abbreviation-pool records.

_IMPLEMENTATION_SAFEGUARDS: tuple[str, ...] = (
    "must_be_predicted_document_type_unknown",
    "must_be_in_insufficient_text_visibility_bucket",
    "must_be_in_language_visibility_unknown_routing_bucket",
    "must_have_medical_abbreviation_shape_detected",
)


def _safeguard_fires(rule: str, record: Mapping[str, Any]) -> bool:
    if rule == "must_be_predicted_document_type_unknown":
        return _g(record, "predicted_document_type") != "unknown"
    if rule == "must_be_in_insufficient_text_visibility_bucket":
        return (
            _g(record, "unknown_failure_bucket")
            != "insufficient_text_visibility"
        )
    if rule == "must_be_in_language_visibility_unknown_routing_bucket":
        return (
            _g(record, "unknown_ocr_routing_bucket")
            != "language_visibility_unknown"
        )
    if rule == "must_have_medical_abbreviation_shape_detected":
        return _g(record, "medical_abbreviation_shape_detected") != "yes"
    return True


def _fails_any_implementation_safeguard(record: Mapping[str, Any]) -> bool:
    return any(_safeguard_fires(r, record) for r in _IMPLEMENTATION_SAFEGUARDS)


# ── Flag plumbing ────────────────────────────────────────────────────────────

def is_latin_abbreviation_metadata_default_disabled() -> bool:
    """Return True. The helper is OFF by default."""
    return True


def is_latin_abbreviation_metadata_enabled(
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return True iff the SEPARATE abbreviation env-gated flag is set."""
    source = env if env is not None else os.environ
    value = str(
        source.get(LATIN_ABBREVIATION_METADATA_ENV_VAR, "")
    ).strip().lower()
    return value in _TRUTHY


def _resolve_enabled(
    enabled: Optional[bool],
    env: Optional[Mapping[str, str]],
) -> bool:
    if enabled is True:
        return True
    if enabled is False:
        return False
    return is_latin_abbreviation_metadata_enabled(env)


# ── Public API ───────────────────────────────────────────────────────────────

def derive_latin_medical_abbreviation_metadata_label(
    record: Mapping[str, Any],
    *,
    enabled: Optional[bool] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Return ``"latin_medical_abbreviation_context"``, or ``None``.

    The function is pure. It never mutates ``record``. It never modifies
    raw detector output. It never auto-accepts, classifies clinical
    meaning, parses or expands the abbreviation, parses lab values /
    medications / doses / DDIs, writes active clinical facts, or changes
    the data-layer document type.
    """
    if not _resolve_enabled(enabled, env):
        return None
    if not matches_positive_signal_pattern(record):
        return None
    if violates_any_exclusion_rule(record):
        return None
    if _fails_any_implementation_safeguard(record):
        return None
    return PROPAGATED_METADATA_LABEL


__all__ = [
    "EXCLUSION_RULES",
    "LATIN_ABBREVIATION_METADATA_DISCLAIMER",
    "LATIN_ABBREVIATION_METADATA_ENV_VAR",
    "LATIN_ABBREVIATION_METADATA_LABEL",
    "POSITIVE_SIGNAL_PATTERN",
    "PROPAGATED_METADATA_DISCLAIMER",
    "PROPAGATED_METADATA_LABEL",
    "derive_latin_medical_abbreviation_metadata_label",
    "is_latin_abbreviation_metadata_default_disabled",
    "is_latin_abbreviation_metadata_enabled",
    "matches_positive_signal_pattern",
    "violates_any_exclusion_rule",
]
