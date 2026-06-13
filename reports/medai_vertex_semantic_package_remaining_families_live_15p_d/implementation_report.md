# MEDAI-VERTEX-SEMANTIC-PACKAGE-COMPARISON-REMAINING-FAMILIES-LIVE-15P-D

- Status: `PASS`
- Families attempted: `3` | passed: `3`
- Live call count: `3` (max 3; one per family; stop on first failure)
- Provider responses received: `3`
- Provider route: `vertex` | Model: `gemini-2.5-flash-lite`
- schema_validation_pass_count: `3`
- source_visible_body_preserved_count: `3`
- evidence_anchor_preserved_count: `3`
- candidate_facts_separated_count: `3`
- unknown_values_explicit_count: `3`
- uncertainty_flags_visible_count: `3`
- hallucinated_field_count: `0`
- posted_body_allowed_top_level_keys_only_count: `3`
- Token totals: prompt=`1137` output=`1037` total=`2174`
- review_required_all: `True` | auto_accept_all_false: `True` | active_written_count: `0`
- Privacy result: `passed` | Billing check pending: `True`

## Safety

- One live Vertex call per remaining family (3 max); no retries; stop on first failure.
- Each posted body contains only `contents` and `generationConfig` (no MedAI metadata).
- JSON-only, temperature=0, maxOutputTokens<=512; no diagnosis/treatment requested.
- No credentials/tokens/auth headers/ADC paths recorded; sanitized responses only.
- No active MKB writes; output remains review-bound; no auto-accept.
- 15P-C portal result-card live call NOT repeated; 15N-R4 smoke NOT rerun.
