"""MEDAI-DOC-TYPE-UNKNOWN-DIAG-12A - Operator surface for Latin medical
abbreviation metadata.

Pure data-only render helper that consumes the DIAG-11A-IMPLEMENTATION
abbreviation helper. Mirrors the DIAG-08A / DIAG-10A pattern but is gated
by a SEPARATE env var:

    MEDAI_DOC_TYPE_LATIN_ABBREVIATION_METADATA_ENABLED

Three-way flag separation
-------------------------
This module participates in a strict 3-way flag separation:

    * ``MEDAI_DOC_TYPE_OPERATOR_REVIEW_BADGE_ENABLED``           (DIAG-07A/08A)
    * ``MEDAI_DOC_TYPE_LANGUAGE_PROPAGATION_METADATA_ENABLED``   (DIAG-09A/10A)
    * ``MEDAI_DOC_TYPE_LATIN_ABBREVIATION_METADATA_ENABLED``     (DIAG-11A/12A)

Each env var toggles exactly one lever. Setting one does NOT activate any
of the others.

What this module does NOT do
----------------------------
* Does NOT parse or expand the abbreviation.
* Does NOT modify raw language-detector output.
* Does NOT change the data-layer document type.
* Does NOT auto-accept.
* Does NOT promote document type.
* Does NOT classify clinical meaning.
* Does NOT parse lab values / medications / doses / frequencies /
  durations / DDIs.
* Does NOT write active clinical facts.
* Does NOT attach any button / form / callback / action handle.
* Does NOT change OCR routing, OCR engine, classifier behavior,
  thresholds, scoring, cue packs, B07, ROUTE-FIX, DB schema, command
  allowlist, or external API behavior.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from clinical_knowledge.document_type.latin_abbreviation_metadata import (
    LATIN_ABBREVIATION_METADATA_ENV_VAR,
    LATIN_ABBREVIATION_METADATA_LABEL,
    derive_latin_medical_abbreviation_metadata_label,
    is_latin_abbreviation_metadata_enabled,
)

# Operator-visible plain-language display text.
LATIN_ABBREVIATION_DISPLAY_TEXT = (
    "metadata: Latin medical abbreviation context - review required"
)

# Snake_case vocab token (safe under the existing privacy regex set).
LATIN_ABBREVIATION_VOCAB_TOKEN = (
    "latin_medical_abbreviation_context_review_required"
)

# Operator-visible disclaimer (extends the DIAG-08A/10A wording with an
# explicit no-parse / no-expand statement, per DIAG-12A spec).
LATIN_ABBREVIATION_UI_DISCLAIMER = (
    "Review metadata only. Not a final document type. "
    "Not clinical interpretation. "
    "Abbreviations are not parsed or expanded."
)

# Plain-language label for the surrounding expander / section.
LATIN_ABBREVIATION_EXPANDER_LABEL = "Latin abbreviation metadata"


def latin_abbreviation_operator_surface_is_enabled(
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """Mirror the abbreviation helper's env-gating predicate so callers can
    cheaply check whether the operator surface should render."""
    return is_latin_abbreviation_metadata_enabled(env)


def render_plan_for_latin_abbreviation(
    record: Mapping[str, Any],
    *,
    enabled: Optional[bool] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Optional[dict[str, Any]]:
    """Return a structured render plan, or ``None``.

    The plan is data-only. The consumer is responsible for translating it
    into widgets. The plan is read-only and carries no operator-action
    handle.
    """
    label = derive_latin_medical_abbreviation_metadata_label(
        record, enabled=enabled, env=env,
    )
    if label is None:
        return None
    return {
        "expander_label": LATIN_ABBREVIATION_EXPANDER_LABEL,
        "markdown_lines": [
            f"**{LATIN_ABBREVIATION_DISPLAY_TEXT}**",
            f"_vocab token:_ `{LATIN_ABBREVIATION_VOCAB_TOKEN}`",
            f"_source label:_ `{label}` "
            f"(via `MEDAI-DOC-TYPE-UNKNOWN-DIAG-11A-IMPLEMENTATION`)",
        ],
        "disclaimer_line": LATIN_ABBREVIATION_UI_DISCLAIMER,
        "is_read_only": True,
        "no_action_attached": True,
        "review_bound": True,
        "is_clinical_classification": False,
        "is_final_document_type": False,
        "is_auto_accept": False,
        "is_active_clinical_fact": False,
        "is_data_layer_document_type_change": False,
        "raw_detector_output_unchanged": True,
        "abbreviation_parsed": False,
        "abbreviation_expanded": False,
        "badge_vocab_token": LATIN_ABBREVIATION_VOCAB_TOKEN,
        "badge_text": LATIN_ABBREVIATION_DISPLAY_TEXT,
        "badge_source_block":
            "MEDAI-DOC-TYPE-UNKNOWN-DIAG-11A-IMPLEMENTATION",
        "badge_env_var": LATIN_ABBREVIATION_METADATA_ENV_VAR,
    }


__all__ = [
    "LATIN_ABBREVIATION_DISPLAY_TEXT",
    "LATIN_ABBREVIATION_EXPANDER_LABEL",
    "LATIN_ABBREVIATION_METADATA_ENV_VAR",
    "LATIN_ABBREVIATION_UI_DISCLAIMER",
    "LATIN_ABBREVIATION_VOCAB_TOKEN",
    "latin_abbreviation_operator_surface_is_enabled",
    "render_plan_for_latin_abbreviation",
]
