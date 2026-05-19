"""MEDAI-DOC-TYPE-UNKNOWN-DIAG-09A-IMPLEMENTATION - Language detector metadata
propagation helper.

This module implements the spec published in UNKNOWN-DIAG-09A (commit
5122d93). It exposes a single pure function:

    derive_language_propagation_metadata_label(record, *, enabled=None, env=None)

The function returns the controlled-vocabulary label

    ``"latin_detector_likely_english_context"``

only when ALL of the following hold:

    1. The propagation flag is explicitly enabled. The flag is plumbed
       via the keyword argument ``enabled=True`` OR the environment
       variable ``MEDAI_DOC_TYPE_LANGUAGE_PROPAGATION_METADATA_ENABLED``
       set to a truthy value (``1`` / ``true`` / ``yes`` / ``on`` /
       ``enabled``). The env var is intentionally DISTINCT from the
       DIAG-07A operator-badge env var so setting one does NOT enable the
       other.
    2. Every field of the 11-field positive signal pattern holds.
    3. None of the 12 exclusion rules fires.
    4. The record does NOT overlap with the numeric-table safe-default
       pool already handled by DIAG-06A / 07A / 08A.

The function is pure. It never mutates the record. It never modifies the
raw language-detector output. It never auto-accepts, classifies clinical
meaning, parses lab values, parses medications / dose / frequency /
duration / DDI, or writes active clinical facts. It never changes the
data-layer document type. Review-bound status is preserved by the caller
because the helper does not touch the record.

Default-off
-----------
The helper is OFF by default. With no kwarg and the env var unset (or
falsy), the function always returns ``None`` and every pipeline that
never imports the module is unaffected. The rollback path is any of:

    * Omit the kwarg AND leave the env var unset.
    * Set the env var to a falsy value (``0`` / ``false`` / ``no`` /
      ``off`` / ``disabled``).
    * Pass ``enabled=False`` explicitly (overrides any env setting).
    * Never import the module.

What this module does NOT do
----------------------------
* Does NOT modify raw language-detector output. The label is a separate
  piece of safe metadata; the caller decides what to do with it.
* Does NOT change the data-layer document type. Records flagged Unknown
  remain Unknown.
* Does NOT auto-accept.
* Does NOT classify clinical meaning.
* Does NOT parse lab values / medications / doses / frequencies /
  durations / DDIs.
* Does NOT write active clinical facts.
* Does NOT change OCR routing, OCR engine, classifier behavior,
  thresholds, scoring, cue packs, B07, ROUTE-FIX, DB schema, command
  allowlist, or external API behavior.
"""
from __future__ import annotations

import os
from typing import Any, Mapping, Optional

from clinical_knowledge.document_type.numeric_table_safe_default import (
    derive_numeric_table_safe_default_label,
)

# ── Controlled-vocabulary constants ──────────────────────────────────────────

PROPAGATED_METADATA_LABEL = "latin_detector_likely_english_context"

# Plain-language operator-facing description (informational; the canonical
# token is the snake_case label above).
PROPAGATED_METADATA_DISCLAIMER = (
    "Safe metadata only. Not a final document type. "
    "Not clinical interpretation. Raw detector output unchanged."
)

# Env var that enables the propagation helper across processes. This is
# INTENTIONALLY distinct from the DIAG-07A operator-badge env var so a
# rollout / rollback of one does not affect the other.
LANGUAGE_PROPAGATION_METADATA_ENV_VAR = (
    "MEDAI_DOC_TYPE_LANGUAGE_PROPAGATION_METADATA_ENABLED"
)

# 11-field positive signal pattern. Each entry is (synthetic_key,
# expected_value_for_documentation).
POSITIVE_SIGNAL_PATTERN: tuple[tuple[str, str], ...] = (
    ("detector_attempted",                       "yes"),
    ("detector_input_bucket",                    "sufficient"),
    ("detector_confidence_bucket",               "high"),
    ("script_detection_result",                  "latin"),
    ("dominant_script",                          "latin"),
    ("language_visibility_status",               "latin_visible_language_unknown"),
    ("detector_output_not_propagated",           "yes"),
    ("alphabetic_ratio_sufficient_for_language", "yes"),
    ("no_cyrillic_dominant_signal",              "yes"),
    ("no_mixed_script_signal",                   "yes"),
    ("no_low_confidence_detector_signal",        "yes"),
)

EXCLUSION_RULES: tuple[str, ...] = (
    "exclude_cyrillic_dominant_records",
    "exclude_mixed_script_records",
    "exclude_low_detector_confidence_records",
    "exclude_insufficient_detector_input_records",
    "exclude_no_text_layer_records",
    "exclude_image_like_but_not_routed_records",
    "exclude_table_heavy_numeric_safe_default_records_already_handled",
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
        return _g(record, "detector_confidence_bucket") == expected
    if key == "script_detection_result":
        return _g(record, "script_detection_result") == expected
    if key == "dominant_script":
        return _g(record, "dominant_script") == expected
    if key == "language_visibility_status":
        return _g(record, "language_visibility_status") == expected
    if key == "detector_output_not_propagated":
        return (
            _g(record, "language_detector_attempted") == "yes"
            and _g(record, "language_detector_input_bucket") == "sufficient"
            and _g(record, "detector_confidence_bucket") in {"medium", "high"}
            and _g(record, "language_visibility_status").startswith(
                "latin_visible_language_unknown"
            )
        )
    if key == "alphabetic_ratio_sufficient_for_language":
        return _g(record, "alphabetic_content_bucket") == "high"
    if key == "no_cyrillic_dominant_signal":
        return _g(record, "dominant_script") != "cyrillic"
    if key == "no_mixed_script_signal":
        return _g(record, "dominant_script") != "mixed"
    if key == "no_low_confidence_detector_signal":
        return _g(record, "detector_confidence_bucket") != "low"
    return False


def matches_positive_signal_pattern(record: Mapping[str, Any]) -> bool:
    """Return True iff every field of the 11-field positive signal pattern holds."""
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
        # If the DIAG-06A helper would label this record, it is already
        # handled upstream and must NOT receive a propagation label here.
        return (
            derive_numeric_table_safe_default_label(record, enabled=True)
            is not None
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
        )
        return any(_g(record, k) == "" for k in required)
    # Unknown rule -> defensively fire (treat as a violation).
    return True


def violates_any_exclusion_rule(record: Mapping[str, Any]) -> bool:
    """Return True iff any of the 12 exclusion rules fires on the record."""
    return any(_exclusion_rule_fires(rule, record) for rule in EXCLUSION_RULES)


# ── Implementation-level safeguards (beyond the 11-field signature) ──────────
#
# The DIAG-09A spec's 11-field positive signal pattern was validated on the
# 11 priority records. On the full 507-row corpus the same 11 fields also
# match many additional records that DIAG-02 / DIAG-03 / DIAG-04 route to
# other levers (numeric-table safe-default, abbreviation, table-header,
# table-heavy policy audit) OR records that are not even Unknown. To satisfy
# the DIAG-09A acceptance criterion
#
#     only_exact_propagation_signature_records_receive_propagated_metadata
#
# the helper enforces a small set of implementation-level safeguards that
# embed the upstream-filter conditions implicit in the priority slice. The
# safeguards never make the helper LESS conservative; they only narrow the
# set of records that can receive the propagated metadata label.

_IMPLEMENTATION_SAFEGUARDS: tuple[str, ...] = (
    "must_be_predicted_document_type_unknown",
    "must_be_in_insufficient_text_visibility_bucket",
    "must_be_in_language_visibility_unknown_routing_bucket",
    "must_have_no_medical_abbreviation_shape_detected",
    "must_not_be_in_table_heavy_diag03_sub_pool",
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
    if rule == "must_have_no_medical_abbreviation_shape_detected":
        return _g(record, "medical_abbreviation_shape_detected") == "yes"
    if rule == "must_not_be_in_table_heavy_diag03_sub_pool":
        # DIAG-03 routes a record to the table-heavy sub-pool when
        # table_like=yes AND (numeric in {medium, high} OR alphabetic in
        # {low, medium}). For the propagation pool we require:
        #   table_like != yes, OR numeric not in {medium, high}.
        # The positive signal already enforces alphabetic_content_bucket=high
        # so the "alphabetic in {low, medium}" branch is not triggered.
        table_like = _g(record, "table_like_structure_detected") == "yes"
        numeric_heavy = _g(record, "numeric_content_bucket") in {
            "medium", "high",
        }
        return table_like and numeric_heavy
    return True


def _fails_any_implementation_safeguard(record: Mapping[str, Any]) -> bool:
    return any(_safeguard_fires(r, record) for r in _IMPLEMENTATION_SAFEGUARDS)


# ── Flag plumbing ────────────────────────────────────────────────────────────

def is_language_propagation_metadata_default_disabled() -> bool:
    """Return True. The helper is OFF by default."""
    return True


def is_language_propagation_metadata_enabled(
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return True iff the SEPARATE env-gated propagation flag is set."""
    source = env if env is not None else os.environ
    value = str(
        source.get(LANGUAGE_PROPAGATION_METADATA_ENV_VAR, "")
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
    return is_language_propagation_metadata_enabled(env)


# ── Public API ───────────────────────────────────────────────────────────────

def derive_language_propagation_metadata_label(
    record: Mapping[str, Any],
    *,
    enabled: Optional[bool] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Return ``PROPAGATED_METADATA_LABEL``, or ``None``.

    :param record:  per-file privacy-safe record (same shape as the
                    FAMILY-04 anonymized per-file table). Must NOT carry
                    parsed lab values, medications, doses, frequencies,
                    or DDI findings.
    :param enabled: explicit override. ``True`` forces the helper on
                    (subject to signature/exclusion/overlap rules).
                    ``False`` forces the helper off. ``None`` defers to
                    the SEPARATE env var
                    ``MEDAI_DOC_TYPE_LANGUAGE_PROPAGATION_METADATA_ENABLED``.
                    The DIAG-07A operator-badge env var does NOT enable
                    this helper.
    :param env:     optional mapping for env-var injection (test hook).

    The function is pure. It never mutates ``record``. It never modifies
    raw detector output. It never auto-accepts, classifies clinical
    meaning, parses values, or writes active clinical facts. The data-
    layer document type is unchanged.
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
    "LANGUAGE_PROPAGATION_METADATA_ENV_VAR",
    "POSITIVE_SIGNAL_PATTERN",
    "PROPAGATED_METADATA_DISCLAIMER",
    "PROPAGATED_METADATA_LABEL",
    "derive_language_propagation_metadata_label",
    "is_language_propagation_metadata_default_disabled",
    "is_language_propagation_metadata_enabled",
    "matches_positive_signal_pattern",
    "violates_any_exclusion_rule",
]
