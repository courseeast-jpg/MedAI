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

__all__ = [
    "DERIVED_LABEL",
    "EXCLUSION_RULES",
    "POSITIVE_SIGNATURE",
    "derive_numeric_table_safe_default_label",
    "is_disabled_by_default",
    "matches_positive_signature",
    "violates_any_exclusion_rule",
]
