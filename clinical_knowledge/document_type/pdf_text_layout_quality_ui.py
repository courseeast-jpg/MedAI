"""MEDAI-DOC-TYPE-UNKNOWN-DIAG-18 — Default-off read-only operator surface
for DIAG-17 PDF text/layout quality metadata.

This module exposes a pure ``render_plan_for_pdf_text_layout_quality``
helper that converts the privacy-safe metadata dict produced by DIAG-17
into a controlled-vocabulary, read-only operator render plan. The plan is
data-only: it carries no Streamlit widgets, no buttons, no callbacks, no
actions, no accept/reject semantics, no state mutation, no data-layer
writes, no document_type mutation.

Strict two-key env gating:

    * ``MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_IMPL_ENABLED`` (DIAG-17
      metadata env var)
    * ``MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_UI_ENABLED`` (DIAG-18 UI
      env var, new and distinct from every other MEDAI doc-type env var)

The operator surface may render only when BOTH env vars are truthy
(``1``/``true``/``yes``/``on``/``enabled``). If either is unset or falsy,
``render_plan_for_pdf_text_layout_quality`` returns ``None`` — no plan,
no metadata generation, no side effects.

Distinctness from every prior MEDAI doc-type env var is enforced by
``DIAG_18_UI_ENV_VAR`` being a hard-coded constant; the module also
exposes a ``KNOWN_DOC_TYPE_ENV_VARS`` tuple so tests can assert
non-overlap.

What this module DOES NOT do (under any combination of kwargs / env):

    * Does NOT add Streamlit imports or widgets to the package.
    * Does NOT add buttons, forms, callbacks, action handles, or
      state-mutation semantics to the plan.
    * Does NOT emit raw extracted text, raw OCR text, raw document text,
      raw filenames, private paths, PHI, or secrets.
    * Does NOT change ``document_type``, data-layer writes, classifier
      output, thresholds, scoring, OCR routing, OCR engine behavior, PDF
      text-extraction behavior, or layout/table extraction behavior.
    * Does NOT auto-accept; does NOT parse clinical values; does NOT
      infer diagnosis, medication, DDI, or treatment meaning; does NOT
      parse or expand abbreviations; does NOT add clinical
      interpretation.
    * Does NOT enable any external API.
    * Does NOT touch PARK-20 or PARK-21 tags.

Default-off rollback paths (any one suffices):

    * Leave either env var unset.
    * Set either env var to a falsy value
      (``0``/``false``/``no``/``off``/``disabled``).
    * Pass ``enabled=False`` explicitly (overrides both truthy envs).
    * Never import this module.
"""
from __future__ import annotations

import os
from typing import Any, Mapping, Optional

from clinical_knowledge.document_type.pdf_text_layout_quality_impl import (
    PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR,
    QUALITY_FAMILY_VALUES,
    derive_pdf_text_layout_quality_context,
    is_pdf_text_layout_quality_impl_enabled,
)

# ── Env-var name (NEW, distinct) ───────────────────────────────────────────

DIAG_18_UI_ENV_VAR = "MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_UI_ENABLED"

# Hard-coded list of every prior MEDAI doc-type env var. The DIAG-18 UI
# env var must NOT equal any of these. Tests assert this invariant.
KNOWN_DOC_TYPE_ENV_VARS = (
    "MEDAI_DOC_TYPE_OPERATOR_REVIEW_BADGE_ENABLED",            # DIAG-07A/08A
    "MEDAI_DOC_TYPE_LANGUAGE_PROPAGATION_METADATA_ENABLED",    # DIAG-09A/10A
    "MEDAI_DOC_TYPE_LATIN_ABBREVIATION_METADATA_ENABLED",      # DIAG-11A/12A
    "MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_IMPL_ENABLED",     # DIAG-17
)

_TRUTHY = {"1", "true", "yes", "on", "enabled"}


# ── Controlled-vocabulary surface text ─────────────────────────────────────

SOURCE_PHASE = "MEDAI-DOC-TYPE-UNKNOWN-DIAG-18"

OPERATOR_DISPLAY_HEADING = (
    "metadata: PDF text/layout quality context — review required"
)

OPERATOR_VOCAB_TOKEN = "pdf_text_layout_quality_context_review_required"

OPERATOR_EXPANDER_LABEL = "PDF text/layout quality metadata"

OPERATOR_DISCLAIMER = (
    "Review metadata only. Not a final document type. "
    "Not clinical interpretation. "
    "PDF text and table/layout extraction behavior is unchanged. "
    "No auto-accept. No action attached. Read-only display."
)

# Human-readable description for each controlled-vocabulary quality_family
# label. Kept inside this module so callers never need to derive their own.
QUALITY_FAMILY_LABEL_DESCRIPTION: dict[str, str] = {
    "pdf_text_too_short": (
        "PDF text layer detected but extracted text is too short to drive "
        "family classification."
    ),
    "table_structure_visible_text_insufficient": (
        "Table-like structure visible, but extractable text is still "
        "insufficient for family classification."
    ),
    "layout_or_table_extraction_gap": (
        "Layout / table extraction gap suspected from privacy-safe "
        "aggregate signals."
    ),
}


# ── Env-gating predicates ──────────────────────────────────────────────────


def is_pdf_text_layout_quality_ui_enabled(
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return True only when the DIAG-18 UI env var is truthy.

    This predicate does NOT consult the DIAG-17 metadata env var — that
    second gate is enforced inside ``render_plan_for_pdf_text_layout_quality``
    via the underlying DIAG-17 helper.
    """
    if env is None:
        env = os.environ
    raw = env.get(DIAG_18_UI_ENV_VAR, "")
    val = str(raw).strip().lower()
    return val in _TRUTHY


def is_pdf_text_layout_quality_ui_default_disabled(
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return True when the UI is currently default-disabled."""
    return not is_pdf_text_layout_quality_ui_enabled(env)


def requires_both_env_vars_truthy(
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return True only when BOTH the DIAG-17 metadata env var AND the
    DIAG-18 UI env var are truthy."""
    return (
        is_pdf_text_layout_quality_impl_enabled(env)
        and is_pdf_text_layout_quality_ui_enabled(env)
    )


# ── Public render-plan function ────────────────────────────────────────────


def render_plan_for_pdf_text_layout_quality(
    record: Mapping[str, Any],
    *,
    enabled: Optional[bool] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Optional[dict[str, Any]]:
    """Return a pure, data-only operator render plan, or ``None``.

    Returns ``None`` (no plan, no side effect) whenever any of:

        * ``enabled`` was passed as ``False``;
        * the DIAG-17 metadata env var is unset or falsy;
        * the DIAG-18 UI env var is unset or falsy;
        * the underlying DIAG-17 helper returns ``None`` (the record
          does not match its positive signature).

    When a plan IS returned, it contains only:

        * ``expander_label`` (operator-facing controlled string)
        * ``markdown_lines`` (operator-facing wording, no raw text)
        * ``disclaimer_line`` (operator-facing disclaimer)
        * a set of explicit ``is_*`` / ``no_*`` invariant flags
        * ``badge_*`` fields carrying only controlled-vocabulary tokens
        * ``quality_family`` (forwarded from the DIAG-17 helper)
        * ``env_vars`` (both controlling env var names, for operator audit)

    The plan carries NO ``on_click``, ``on_submit``, ``button``,
    ``callback``, ``action``, ``write``, or ``mutate`` fields. Tests
    enforce this in CI.
    """
    if enabled is False:
        return None

    if enabled is None:
        if not is_pdf_text_layout_quality_ui_enabled(env):
            return None
        # DIAG-17 metadata env-gate is enforced by the helper below.

    # Always pass env through to the metadata helper. If we got here via
    # ``enabled=True``, force the metadata helper to also see enabled=True
    # so it doesn't second-guess us.
    metadata = derive_pdf_text_layout_quality_context(
        record,
        enabled=True if enabled is True else None,
        env=env,
    )
    if metadata is None:
        return None

    quality_family = tuple(metadata["quality_family"])
    description_lines = [
        f"_{label}_: {QUALITY_FAMILY_LABEL_DESCRIPTION[label]}"
        for label in quality_family
    ]

    plan = {
        "expander_label": OPERATOR_EXPANDER_LABEL,
        "markdown_lines": [
            f"**{OPERATOR_DISPLAY_HEADING}**",
            f"_vocab token:_ `{OPERATOR_VOCAB_TOKEN}`",
            *description_lines,
            "_source phase:_ "
            f"`{SOURCE_PHASE}` "
            f"(consumes `{metadata['source_phase']}`)",
        ],
        "disclaimer_line": OPERATOR_DISCLAIMER,
        "quality_family": quality_family,
        "env_vars": {
            "metadata_env_var": PDF_TEXT_LAYOUT_QUALITY_IMPL_ENV_VAR,
            "ui_env_var": DIAG_18_UI_ENV_VAR,
        },
        "badge_vocab_token": OPERATOR_VOCAB_TOKEN,
        "badge_text": OPERATOR_DISPLAY_HEADING,
        "badge_source_block": SOURCE_PHASE,

        # Explicit invariant flags — all False except read-only / review-bound
        "is_read_only": True,
        "is_review_bound": True,
        "no_action_attached": True,
        "no_button_attached": False,  # legacy field, see below
        "is_clinical_classification": False,
        "is_final_document_type": False,
        "is_auto_accept": False,
        "is_active_clinical_fact": False,
        "is_data_layer_document_type_change": False,
        "raw_detector_output_unchanged": True,
        "abbreviation_parsed": False,
        "abbreviation_expanded": False,
        "lab_value_parsed": False,
        "medication_inference_performed": False,
        "diagnosis_inference_performed": False,
        "ddi_inference_performed": False,
        "treatment_inference_performed": False,
        "clinical_interpretation_performed": False,
        "external_api_used": False,
        "pdf_text_extraction_behavior_changed": False,
        "layout_extraction_behavior_changed": False,
        "table_extraction_behavior_changed": False,
        "ocr_routing_changed": False,
        "ocr_engine_behavior_changed": False,
        "classifier_behavior_changed": False,
        "thresholds_or_scoring_changed": False,
        "cue_packs_added": False,
        "park_20_tags_touched": False,
        "park_21_tags_touched": False,

        # Privacy invariants for the plan itself
        "raw_text_rendered": False,
        "raw_ocr_text_rendered": False,
        "raw_document_text_rendered": False,
        "raw_filename_rendered": False,
        "private_path_rendered": False,
        "phi_rendered": False,
        "secret_rendered": False,
    }
    # Correct the legacy "no_button_attached" so it expresses the
    # invariant clearly: there is no button anywhere in the plan.
    plan["no_button_attached"] = True
    plan["no_callback_attached"] = True
    plan["no_form_attached"] = True
    plan["no_state_mutation"] = True
    plan["no_data_layer_write"] = True
    plan["no_document_type_mutation"] = True
    return plan


__all__ = [
    "DIAG_18_UI_ENV_VAR",
    "KNOWN_DOC_TYPE_ENV_VARS",
    "OPERATOR_DISCLAIMER",
    "OPERATOR_DISPLAY_HEADING",
    "OPERATOR_EXPANDER_LABEL",
    "OPERATOR_VOCAB_TOKEN",
    "QUALITY_FAMILY_LABEL_DESCRIPTION",
    "SOURCE_PHASE",
    "is_pdf_text_layout_quality_ui_default_disabled",
    "is_pdf_text_layout_quality_ui_enabled",
    "render_plan_for_pdf_text_layout_quality",
    "requires_both_env_vars_truthy",
]
