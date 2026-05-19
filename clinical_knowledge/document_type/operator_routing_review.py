"""MEDAI-DOC-TYPE-UNKNOWN-DIAG-07A - Operator Routing Review Integration.

Thin, default-off consumer of the DIAG-06A helper for the operator routing-
review surface only. Adds:

    derive_operator_review_badge(record, *, enabled=None, env=None) -> dict | None

The function returns a structured operator-review badge ONLY when:

    1. The operator-review flag is explicitly enabled (either via the
       ``enabled=True`` argument, or via the
       ``MEDAI_DOC_TYPE_OPERATOR_REVIEW_BADGE_ENABLED`` environment variable
       set to a truthy value such as "1" / "true" / "yes" / "on"), AND
    2. The DIAG-06A helper returns the safe-default label (which itself
       requires the exact 14-field positive signature, no exclusion-rule
       violation, and no implementation-level safeguard failure).

In every other case the function returns ``None``. The default outside this
explicit operator-review call site is OFF. The disable / rollback path is
simply to omit ``enabled=True`` AND keep the environment variable unset.

The badge is review-bound metadata only. It carries explicit fields that
flag it as NOT a clinical classification and NOT a final document type so
any consuming UI can render it as a read-only review hint.

Hard guardrails preserved
-------------------------
* No auto-accept.
* No clinical interpretation.
* No lab-value parsing.
* No medication / dose / frequency / duration / DDI parsing.
* No active clinical fact writes.
* All affected documents remain review-bound.

The DIAG-06A helper remains default-off outside this explicit operator-
review call site; existing pipelines that never invoke
``derive_operator_review_badge`` are unaffected.
"""
from __future__ import annotations

import os
from typing import Any, Mapping, Optional

from clinical_knowledge.document_type.numeric_table_safe_default import (
    DERIVED_LABEL as _NUMERIC_TABLE_DERIVED_LABEL,
    derive_numeric_table_safe_default_label,
)

# ── Controlled-vocabulary constants ──────────────────────────────────────────

# Snake_case identifier used in machine-readable contexts. Safe under the
# repo's privacy regex set (no "policy <5+ char word>" pattern).
OPERATOR_REVIEW_BADGE_VOCAB_TOKEN = "table_latin_likely_english_context_review_required"

# Plain-language operator-facing badge text. Designed to be unambiguous about
# what the badge IS (a routing-review hint) and what it IS NOT (a clinical
# classification or a final document type).
OPERATOR_REVIEW_BADGE_TEXT = (
    "metadata: table-Latin-likely-English-context - review required"
)

# Operator UI guidance text. The UI must render this verbatim with the badge.
OPERATOR_REVIEW_BADGE_DISCLAIMER = (
    "This is a routing-review hint for the operator queue. It is not a "
    "clinical classification, not a final document type, and does not auto-"
    "accept. The document remains review-bound."
)

# Environment variable that controls the badge across processes when the
# caller does not pass ``enabled=True``. Default is unset / disabled.
OPERATOR_REVIEW_BADGE_ENV_VAR = "MEDAI_DOC_TYPE_OPERATOR_REVIEW_BADGE_ENABLED"

# Module-level default state. Used by ``is_operator_review_badge_default_disabled``.
_DEFAULT_DISABLED = True

_TRUTHY = frozenset({"1", "true", "yes", "on", "enabled"})


# ── Flag plumbing ────────────────────────────────────────────────────────────

def is_operator_review_badge_default_disabled() -> bool:
    """Return True. The badge integration is OFF by default."""
    return _DEFAULT_DISABLED


def is_operator_review_badge_enabled(
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return True iff the environment-gated operator-review flag is set.

    ``env`` may be passed for test injection; defaults to ``os.environ``.
    """
    source = env if env is not None else os.environ
    value = str(source.get(OPERATOR_REVIEW_BADGE_ENV_VAR, "")).strip().lower()
    return value in _TRUTHY


def _resolve_enabled(
    enabled: Optional[bool],
    env: Optional[Mapping[str, str]],
) -> bool:
    if enabled is True:
        return True
    if enabled is False:
        return False
    # enabled is None: consult the env-gated flag.
    return is_operator_review_badge_enabled(env)


# ── Public API ───────────────────────────────────────────────────────────────

def derive_operator_review_badge(
    record: Mapping[str, Any],
    *,
    enabled: Optional[bool] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Optional[dict[str, Any]]:
    """Return the operator-review badge metadata, or ``None``.

    :param record:  per-file privacy-safe record (same shape as the
                    FAMILY-04 anonymized per-file table).
    :param enabled: explicit override. ``True`` forces the badge on for the
                    current call (subject to the DIAG-06A signature/exclusion
                    rules). ``False`` forces the badge off. ``None`` defers
                    to the environment-gated flag.
    :param env:     optional mapping for env-var injection (test hook).

    The returned dict has the shape:

        {
            "badge_vocab_token":          "table_latin_likely_english_context_review_required",
            "badge_text":                 "<plain language operator label>",
            "badge_disclaimer":           "<operator UI guidance>",
            "review_bound":               True,
            "is_clinical_classification": False,
            "is_final_document_type":     False,
            "is_auto_accept":             False,
            "is_active_clinical_fact":    False,
            "source_label":               "latin_script_likely_english_table_context",
            "source_block":               "MEDAI-DOC-TYPE-UNKNOWN-DIAG-06A-IMPLEMENTATION",
        }

    The function is pure. It never mutates ``record``. It never modifies the
    raw language-detector output. It never auto-accepts, classifies clinical
    meaning, parses values, or writes active clinical facts.
    """
    if not _resolve_enabled(enabled, env):
        return None

    underlying = derive_numeric_table_safe_default_label(record, enabled=True)
    if underlying is None:
        return None

    return {
        "badge_vocab_token":          OPERATOR_REVIEW_BADGE_VOCAB_TOKEN,
        "badge_text":                 OPERATOR_REVIEW_BADGE_TEXT,
        "badge_disclaimer":           OPERATOR_REVIEW_BADGE_DISCLAIMER,
        "review_bound":               True,
        "is_clinical_classification": False,
        "is_final_document_type":     False,
        "is_auto_accept":             False,
        "is_active_clinical_fact":    False,
        "source_label":               _NUMERIC_TABLE_DERIVED_LABEL,
        "source_block":               "MEDAI-DOC-TYPE-UNKNOWN-DIAG-06A-IMPLEMENTATION",
    }


__all__ = [
    "OPERATOR_REVIEW_BADGE_DISCLAIMER",
    "OPERATOR_REVIEW_BADGE_ENV_VAR",
    "OPERATOR_REVIEW_BADGE_TEXT",
    "OPERATOR_REVIEW_BADGE_VOCAB_TOKEN",
    "derive_operator_review_badge",
    "is_operator_review_badge_default_disabled",
    "is_operator_review_badge_enabled",
]
