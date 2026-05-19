"""Document-type subpackage: narrow, opt-in helpers.

Currently houses the MEDAI-DOC-TYPE-UNKNOWN-DIAG-06A-IMPLEMENTATION helper:
a small pure function that derives a safe metadata/routing label
(``latin_script_likely_english_table_context``) for records that match a
strict 14-field positive signature and violate none of the 10 exclusion
rules. The helper is default-off; callers must explicitly pass
``enabled=True`` to activate it. It never modifies raw detector output,
never auto-accepts, never classifies clinical meaning, never parses
values, and never writes active clinical facts.
"""
from clinical_knowledge.document_type.numeric_table_safe_default import (
    DERIVED_LABEL,
    EXCLUSION_RULES,
    POSITIVE_SIGNATURE,
    derive_numeric_table_safe_default_label,
    is_disabled_by_default,
    matches_positive_signature,
    violates_any_exclusion_rule,
)
from clinical_knowledge.document_type.operator_routing_review import (
    OPERATOR_REVIEW_BADGE_DISCLAIMER,
    OPERATOR_REVIEW_BADGE_ENV_VAR,
    OPERATOR_REVIEW_BADGE_TEXT,
    OPERATOR_REVIEW_BADGE_VOCAB_TOKEN,
    derive_operator_review_badge,
    is_operator_review_badge_default_disabled,
    is_operator_review_badge_enabled,
)

__all__ = [
    "DERIVED_LABEL",
    "EXCLUSION_RULES",
    "POSITIVE_SIGNATURE",
    "OPERATOR_REVIEW_BADGE_DISCLAIMER",
    "OPERATOR_REVIEW_BADGE_ENV_VAR",
    "OPERATOR_REVIEW_BADGE_TEXT",
    "OPERATOR_REVIEW_BADGE_VOCAB_TOKEN",
    "derive_numeric_table_safe_default_label",
    "derive_operator_review_badge",
    "is_disabled_by_default",
    "is_operator_review_badge_default_disabled",
    "is_operator_review_badge_enabled",
    "matches_positive_signature",
    "violates_any_exclusion_rule",
]
