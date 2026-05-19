"""MEDAI-DOC-TYPE-UNKNOWN-DIAG-10A - Operator surface for propagated language metadata.

Pure data-only render helper that consumes the DIAG-09A-IMPLEMENTATION
propagation helper and produces a structured render-plan dict for the
operator routing-review surface. Mirrors the DIAG-08A pattern but is
gated by a SEPARATE env var:

    MEDAI_DOC_TYPE_LANGUAGE_PROPAGATION_METADATA_ENABLED

The DIAG-07A operator-badge env var (``MEDAI_DOC_TYPE_OPERATOR_REVIEW_BADGE_ENABLED``)
does NOT enable this display, and vice versa - each lever toggles
independently so it can be rolled forward or rolled back on its own.

What this module does NOT do
----------------------------
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

from clinical_knowledge.document_type.language_propagation_metadata import (
    LANGUAGE_PROPAGATION_METADATA_ENV_VAR,
    PROPAGATED_METADATA_LABEL,
    derive_language_propagation_metadata_label,
    is_language_propagation_metadata_enabled,
)

# Operator-visible plain-language display text.
LANGUAGE_PROPAGATION_DISPLAY_TEXT = (
    "metadata: Latin detector likely-English context - review required"
)

# Snake_case vocab token (safe under the existing privacy regex set).
LANGUAGE_PROPAGATION_VOCAB_TOKEN = (
    "latin_detector_likely_english_context_review_required"
)

# Operator-visible disclaimer (verbatim per spec, identical to DIAG-08A's).
LANGUAGE_PROPAGATION_DISCLAIMER = (
    "Review metadata only. Not a final document type. "
    "Not clinical interpretation."
)

# Plain-language label for the surrounding expander / section.
LANGUAGE_PROPAGATION_EXPANDER_LABEL = "Language propagation metadata"


def language_propagation_operator_surface_is_enabled(
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """Mirror the propagation helper's env-gating predicate so callers can
    cheaply check whether the operator surface should render."""
    return is_language_propagation_metadata_enabled(env)


def render_plan_for_language_propagation(
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
    label = derive_language_propagation_metadata_label(
        record, enabled=enabled, env=env,
    )
    if label is None:
        return None
    return {
        "expander_label": LANGUAGE_PROPAGATION_EXPANDER_LABEL,
        "markdown_lines": [
            f"**{LANGUAGE_PROPAGATION_DISPLAY_TEXT}**",
            f"_vocab token:_ `{LANGUAGE_PROPAGATION_VOCAB_TOKEN}`",
            f"_source label:_ `{label}` "
            f"(via `MEDAI-DOC-TYPE-UNKNOWN-DIAG-09A-IMPLEMENTATION`)",
        ],
        "disclaimer_line": LANGUAGE_PROPAGATION_DISCLAIMER,
        "is_read_only": True,
        "no_action_attached": True,
        "review_bound": True,
        "is_clinical_classification": False,
        "is_final_document_type": False,
        "is_auto_accept": False,
        "is_active_clinical_fact": False,
        "is_data_layer_document_type_change": False,
        "raw_detector_output_unchanged": True,
        "badge_vocab_token": LANGUAGE_PROPAGATION_VOCAB_TOKEN,
        "badge_text": LANGUAGE_PROPAGATION_DISPLAY_TEXT,
        "badge_source_block":
            "MEDAI-DOC-TYPE-UNKNOWN-DIAG-09A-IMPLEMENTATION",
        "badge_env_var": LANGUAGE_PROPAGATION_METADATA_ENV_VAR,
    }


__all__ = [
    "LANGUAGE_PROPAGATION_DISCLAIMER",
    "LANGUAGE_PROPAGATION_DISPLAY_TEXT",
    "LANGUAGE_PROPAGATION_EXPANDER_LABEL",
    "LANGUAGE_PROPAGATION_METADATA_ENV_VAR",
    "LANGUAGE_PROPAGATION_VOCAB_TOKEN",
    "language_propagation_operator_surface_is_enabled",
    "render_plan_for_language_propagation",
]
