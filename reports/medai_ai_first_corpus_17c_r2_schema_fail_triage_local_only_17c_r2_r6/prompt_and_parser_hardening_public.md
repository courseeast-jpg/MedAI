# Prompt & parser hardening (17C-R2-R6)

- safe_normalizer_added: `True`
- prompt_hardened_for_strict_json: `True`
- json_mode_enabled_if_supported: `True`
- schema_weakened: `False` | partial_json_accepted: `False` | missing_fields_inferred: `False`
- normalizer_synthetic_replay_all_pass: `True`

The normalizer strips ONE clean Markdown fence around a single JSON object and
rejects prose-wrapped, multiple-object, truncated, and empty responses. The schema
is unchanged; full field + privacy checks still apply to the normalized object.
