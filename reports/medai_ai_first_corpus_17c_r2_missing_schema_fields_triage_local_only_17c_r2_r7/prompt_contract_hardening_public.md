# Prompt contract hardening (17C-R2-R7)

- prompt_hardened_for_required_fields: `True`
- schema_skeleton_instruction_added: `True`
- required_fields_precheck_added: `True`
- schema_weakened: `False` | required_fields_made_optional: `False`
- clinical_values_inferred: `False` | missing_required_values_synthesized: `False`

The prompt now lists every required top-level key and requires the model to emit
all of them (empty arrays for lists, null only where allowed). The 17C-R2 validator
requires ALL core top-level keys and never accepts incomplete output. Nothing is
inferred or synthesized; the schema is not weakened.
