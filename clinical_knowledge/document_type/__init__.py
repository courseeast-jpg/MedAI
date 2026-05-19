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
from clinical_knowledge.document_type.operator_badge_ui import (
    OPERATOR_BADGE_UI_DISCLAIMER,
    OPERATOR_BADGE_UI_EXPANDER_LABEL,
    operator_badge_ui_is_enabled,
    render_plan_for_operator_badge,
)
from clinical_knowledge.document_type.language_propagation_metadata import (
    LANGUAGE_PROPAGATION_METADATA_ENV_VAR,
    PROPAGATED_METADATA_DISCLAIMER,
    PROPAGATED_METADATA_LABEL,
    derive_language_propagation_metadata_label,
    is_language_propagation_metadata_default_disabled,
    is_language_propagation_metadata_enabled,
)
from clinical_knowledge.document_type.language_propagation_operator_surface import (
    LANGUAGE_PROPAGATION_DISCLAIMER,
    LANGUAGE_PROPAGATION_DISPLAY_TEXT,
    LANGUAGE_PROPAGATION_EXPANDER_LABEL,
    LANGUAGE_PROPAGATION_VOCAB_TOKEN,
    language_propagation_operator_surface_is_enabled,
    render_plan_for_language_propagation,
)
from clinical_knowledge.document_type.latin_abbreviation_metadata import (
    LATIN_ABBREVIATION_METADATA_DISCLAIMER,
    LATIN_ABBREVIATION_METADATA_ENV_VAR,
    LATIN_ABBREVIATION_METADATA_LABEL,
    derive_latin_medical_abbreviation_metadata_label,
    is_latin_abbreviation_metadata_default_disabled,
    is_latin_abbreviation_metadata_enabled,
)

__all__ = [
    "DERIVED_LABEL",
    "EXCLUSION_RULES",
    "POSITIVE_SIGNATURE",
    "LANGUAGE_PROPAGATION_DISCLAIMER",
    "LANGUAGE_PROPAGATION_DISPLAY_TEXT",
    "LANGUAGE_PROPAGATION_EXPANDER_LABEL",
    "LANGUAGE_PROPAGATION_METADATA_ENV_VAR",
    "LANGUAGE_PROPAGATION_VOCAB_TOKEN",
    "LATIN_ABBREVIATION_METADATA_DISCLAIMER",
    "LATIN_ABBREVIATION_METADATA_ENV_VAR",
    "LATIN_ABBREVIATION_METADATA_LABEL",
    "OPERATOR_BADGE_UI_DISCLAIMER",
    "OPERATOR_BADGE_UI_EXPANDER_LABEL",
    "OPERATOR_REVIEW_BADGE_DISCLAIMER",
    "OPERATOR_REVIEW_BADGE_ENV_VAR",
    "OPERATOR_REVIEW_BADGE_TEXT",
    "OPERATOR_REVIEW_BADGE_VOCAB_TOKEN",
    "PROPAGATED_METADATA_DISCLAIMER",
    "PROPAGATED_METADATA_LABEL",
    "derive_language_propagation_metadata_label",
    "derive_latin_medical_abbreviation_metadata_label",
    "derive_numeric_table_safe_default_label",
    "derive_operator_review_badge",
    "is_disabled_by_default",
    "is_language_propagation_metadata_default_disabled",
    "is_language_propagation_metadata_enabled",
    "is_latin_abbreviation_metadata_default_disabled",
    "is_latin_abbreviation_metadata_enabled",
    "is_operator_review_badge_default_disabled",
    "is_operator_review_badge_enabled",
    "language_propagation_operator_surface_is_enabled",
    "matches_positive_signature",
    "operator_badge_ui_is_enabled",
    "render_plan_for_language_propagation",
    "render_plan_for_operator_badge",
    "violates_any_exclusion_rule",
]
