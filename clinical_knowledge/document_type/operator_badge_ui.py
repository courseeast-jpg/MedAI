"""MEDAI-DOC-TYPE-UNKNOWN-DIAG-08A - Read-only operator-badge render plan.

This module exposes a single pure function:

    render_plan_for_operator_badge(record, *, enabled=None, env=None) -> dict | None

Returns a structured render-plan dict when the operator-review flag is
enabled AND the underlying DIAG-06A/07A helper returns the safe-default
label for the record, otherwise ``None``.

The function is intentionally Streamlit-free. It carries no UI calls and
no operator-action callbacks. The consumer (e.g. ``app/main.py`` inside the
existing "Advanced technical details" expander) is responsible for turning
the plan's markdown lines into widgets. The plan never carries:

    * a button / form / callback handle
    * a clinical classification or final document type
    * any auto-accept / state-mutation hint
    * any active clinical fact

Default-off behavior is preserved: when the env var
``MEDAI_DOC_TYPE_OPERATOR_REVIEW_BADGE_ENABLED`` is unset / falsy AND the
caller does not pass ``enabled=True``, the function returns ``None`` and
the UI surface remains visually unchanged.

Rollback paths
--------------
* Omit the kwarg AND leave the env var unset (the default).
* Pass ``enabled=False`` explicitly.
* Set the env var to a falsy value (``0`` / ``false`` / ``no`` / ``off`` /
  ``disabled`` / unset).
* Never import the module - existing UI is unaffected.

Hard guardrails preserved
-------------------------
* Read-only metadata badge. No buttons / actions.
* No auto-accept.
* No clinical interpretation.
* No lab-value parsing.
* No medication / dose / frequency / duration / DDI parsing.
* No active clinical fact writes.
* All affected documents remain review-bound.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from clinical_knowledge.document_type.operator_routing_review import (
    OPERATOR_REVIEW_BADGE_DISCLAIMER,
    OPERATOR_REVIEW_BADGE_ENV_VAR,
    OPERATOR_REVIEW_BADGE_TEXT,
    OPERATOR_REVIEW_BADGE_VOCAB_TOKEN,
    derive_operator_review_badge,
    is_operator_review_badge_enabled,
)

# Operator-visible disclaimer required by DIAG-08A spec.
OPERATOR_BADGE_UI_DISCLAIMER = (
    "Review metadata only. Not a final document type. "
    "Not clinical interpretation."
)

# Plain-language label for the surrounding expander / section.
OPERATOR_BADGE_UI_EXPANDER_LABEL = "Operator review metadata"


def render_plan_for_operator_badge(
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
    badge = derive_operator_review_badge(record, enabled=enabled, env=env)
    if badge is None:
        return None
    return {
        "expander_label": OPERATOR_BADGE_UI_EXPANDER_LABEL,
        "markdown_lines": [
            f"**{badge['badge_text']}**",
            f"_vocab token:_ `{badge['badge_vocab_token']}`",
            (
                f"_source:_ `{badge['source_label']}` "
                f"(via `{badge['source_block']}`)"
            ),
        ],
        "disclaimer_line": OPERATOR_BADGE_UI_DISCLAIMER,
        "is_read_only": True,
        "no_action_attached": True,
        "review_bound": True,
        "is_clinical_classification": False,
        "is_final_document_type": False,
        "is_auto_accept": False,
        "is_active_clinical_fact": False,
        "badge_vocab_token": badge["badge_vocab_token"],
        "badge_text": badge["badge_text"],
        "badge_source_block": badge["source_block"],
    }


def operator_badge_ui_is_enabled(
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """Convenience wrapper that exposes the same env-gating used by DIAG-07A."""
    return is_operator_review_badge_enabled(env)


__all__ = [
    "OPERATOR_BADGE_UI_DISCLAIMER",
    "OPERATOR_BADGE_UI_EXPANDER_LABEL",
    "OPERATOR_REVIEW_BADGE_ENV_VAR",
    "operator_badge_ui_is_enabled",
    "render_plan_for_operator_badge",
]
