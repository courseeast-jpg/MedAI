"""MEDAI-DOC-TYPE-UNKNOWN-DIAG-17 — Default-off PDF text/layout quality
metadata helper.

Implements the first default-off implementation pass under the DIAG-16
acceptance criteria for the 21-record residual text-layer scope (11
Sub-track A "text_layer_too_short" + 10 Sub-track B
"table_structure_visible_but_text_insufficient").

The helper is a pure function. With no kwarg and the SEPARATE env var
unset (or falsy), the function always returns ``None`` — meaning callers
that already exist receive no metadata and observe no behavior change.
The new env var is INTENTIONALLY DISTINCT from every prior MEDAI doc-type
env var:

    * ``MEDAI_DOC_TYPE_OPERATOR_REVIEW_BADGE_ENABLED``         (DIAG-07A/08A)
    * ``MEDAI_DOC_TYPE_LANGUAGE_PROPAGATION_METADATA_ENABLED`` (DIAG-09A/10A)
    * ``MEDAI_DOC_TYPE_LATIN_ABBREVIATION_METADATA_ENABLED``   (DIAG-11A/12A)
    * ``MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_IMPL_ENABLED``  (DIAG-17, this module)

What this helper DOES (only when explicitly enabled AND record matches a
narrow positive signature):

    * Returns a small, controlled-vocabulary metadata dict tagging the
      record's quality family. Three controlled-vocabulary values are
      emitted, picked from a fixed alternation:
          - ``pdf_text_too_short``
          - ``table_structure_visible_text_insufficient``
          - ``layout_or_table_extraction_gap``
    * Carries explicit review-required and no-auto-accept flags.
    * Carries explicit "clinical interpretation not performed" flag.

What this helper does NOT do (under any combination of kwargs / env):

    * Does NOT alter the record (no mutation, no in-place writes).
    * Does NOT change document_type at the data layer.
    * Does NOT change classifier, thresholds, scoring, or cue packs.
    * Does NOT change OCR routing or OCR engine behavior.
    * Does NOT change PDF text extraction or layout/table extraction.
    * Does NOT auto-accept.
    * Does NOT parse lab values, medications, dose, frequency, duration,
      DDI, or abbreviations.
    * Does NOT infer diagnosis, medication, DDI, or treatment meaning.
    * Does NOT enable any external API.
    * Does NOT touch PARK-20 or PARK-21 tags.
    * Does NOT emit raw extracted text, raw OCR text, raw document text,
      raw filenames, private paths, PHI, or secrets.

Default-off rollback paths (any one suffices):

    * Omit the ``enabled`` kwarg AND leave the env var unset.
    * Set the env var to a falsy value (``0``/``false``/``no``/``off``/
      ``disabled``).
    * Pass ``enabled=False`` explicitly (overrides a truthy env).
    * Never import the module.
"""
from __future__ import annotations

import os
from typing import Any, Mapping, Optional

# ── Env-var name and default-off semantics ─────────────────────────────────

PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR = (
    "MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_IMPL_ENABLED"
)

_TRUTHY = {"1", "true", "yes", "on", "enabled"}
_FALSY = {"", "0", "false", "no", "off", "disabled"}


def is_pdf_text_layout_quality_impl_default_disabled(
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return True when the helper is currently default-disabled."""
    return not is_pdf_text_layout_quality_impl_enabled(env)


def is_pdf_text_layout_quality_impl_enabled(
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """Pure env-gating predicate. ``env`` defaults to ``os.environ``."""
    if env is None:
        env = os.environ
    raw = env.get(PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR, "")
    val = str(raw).strip().lower()
    if val in _TRUTHY:
        return True
    return False


# ── Controlled-vocabulary outputs ──────────────────────────────────────────

SOURCE_PHASE = "MEDAI-DOC-TYPE-UNKNOWN-DIAG-17"

QUALITY_FAMILY_VALUES = (
    "pdf_text_too_short",
    "table_structure_visible_text_insufficient",
    "layout_or_table_extraction_gap",
)

DISCLAIMER = (
    "Default-off DIAG-17 PDF text/layout quality metadata. Review required. "
    "Not a final document type. Not clinical interpretation. No abbreviation "
    "parsing or expansion. No lab/medication/dose/frequency/duration/DDI "
    "parsing. No auto-accept."
)


# ── Positive signature ─────────────────────────────────────────────────────


def matches_pdf_text_layout_quality_signature(
    record: Mapping[str, Any],
) -> bool:
    """Strict positive signature derived from DIAG-13A/14/15/15B aggregates.

    Returns True only when the record's privacy-safe signals describe a
    text-layer-present-but-too-short case (sub-track A) OR a
    table-structure-visible-but-text-insufficient case (sub-track B).
    """
    pdf_text_layer = record.get("pdf_text_layer_detected")
    image_like = record.get("image_like_pdf")
    alpha = record.get("alphabetic_content_bucket")
    length = record.get("native_text_length_bucket")
    table_like = record.get("table_like_structure_detected")

    if pdf_text_layer != "yes":
        return False
    if image_like != "no":
        return False
    if alpha != "high":
        return False
    if length not in ("none", "tiny", "short"):
        return False
    if table_like not in ("yes", "no"):
        return False
    return True


def _classify_quality_family(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Map a matching record to one or more controlled-vocabulary
    quality-family labels. Multi-label is allowed; the function never
    emits a label outside ``QUALITY_FAMILY_VALUES``.
    """
    table_like = record.get("table_like_structure_detected")
    if table_like == "yes":
        return (
            "table_structure_visible_text_insufficient",
            "layout_or_table_extraction_gap",
        )
    # table_like == "no": text layer is short but no table-like structure
    return ("pdf_text_too_short",)


# ── Public derive function ─────────────────────────────────────────────────


def derive_pdf_text_layout_quality_context(
    record: Mapping[str, Any],
    *,
    enabled: Optional[bool] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Optional[dict[str, Any]]:
    """Return a privacy-safe controlled-vocabulary metadata dict, or ``None``.

    The function is pure: ``record`` is never mutated. The return value is
    a fresh dict that contains only:

        * ``enabled`` (bool, always True when a non-None dict is returned)
        * ``source_phase`` (controlled string)
        * ``quality_family`` (tuple of one or more controlled-vocabulary
          labels from ``QUALITY_FAMILY_VALUES``)
        * ``review_required`` (always True)
        * ``auto_accept_allowed`` (always False)
        * ``clinical_interpretation_performed`` (always False)
        * ``env_var`` (the controlling env-var name, for operator audit)
        * ``disclaimer`` (operator-visible disclaimer)
        * ``raw_text_emitted`` / ``raw_ocr_text_emitted`` /
          ``raw_filename_emitted`` / ``private_path_emitted`` /
          ``phi_emitted`` / ``secret_emitted`` (all always False)

    Default-off behavior: if ``enabled`` is not explicitly True AND the
    env var is unset or falsy, the function returns ``None`` regardless
    of the record's contents.
    """
    if enabled is False:
        return None
    if enabled is None:
        if not is_pdf_text_layout_quality_impl_enabled(env):
            return None
    # enabled is True, OR env says enabled
    if not matches_pdf_text_layout_quality_signature(record):
        return None
    quality_family = _classify_quality_family(record)
    return {
        "enabled": True,
        "source_phase": SOURCE_PHASE,
        "quality_family": quality_family,
        "review_required": True,
        "auto_accept_allowed": False,
        "clinical_interpretation_performed": False,
        "diagnosis_inference_performed": False,
        "medication_inference_performed": False,
        "ddi_inference_performed": False,
        "treatment_inference_performed": False,
        "abbreviation_parsed": False,
        "abbreviation_expanded": False,
        "lab_value_parsed": False,
        "ocr_routing_changed": False,
        "ocr_engine_behavior_changed": False,
        "pdf_text_extraction_behavior_changed": False,
        "layout_extraction_behavior_changed": False,
        "table_extraction_behavior_changed": False,
        "classifier_behavior_changed": False,
        "thresholds_or_scoring_changed": False,
        "cue_packs_added": False,
        "external_api_used": False,
        "data_layer_document_type_changed": False,
        "raw_language_detector_output_changed": False,
        "park_20_tags_touched": False,
        "park_21_tags_touched": False,
        "raw_text_emitted": False,
        "raw_ocr_text_emitted": False,
        "raw_filename_emitted": False,
        "private_path_emitted": False,
        "phi_emitted": False,
        "secret_emitted": False,
        "env_var": PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR,
        "disclaimer": DISCLAIMER,
    }


__all__ = [
    "DISCLAIMER",
    "PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR",
    "QUALITY_FAMILY_VALUES",
    "SOURCE_PHASE",
    "derive_pdf_text_layout_quality_context",
    "is_pdf_text_layout_quality_impl_default_disabled",
    "is_pdf_text_layout_quality_impl_enabled",
    "matches_pdf_text_layout_quality_signature",
]
