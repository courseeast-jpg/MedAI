# MEDAI-VERTEX-SEMANTIC-REVIEW-DECISION-AUDIT-EXPORT-15S

- Decision records loaded (read-only from 15R store): `15`
- Exported JSON / CSV: `15` / `15`
- Per-family: `{'cytology_pathology_narrative': 3, 'mixed_narrative_numeric_result': 4, 'portal_result_cards': 4, 'urinalysis_table_like_lab': 4}`
- Per-action: accepted_for_review `13`, rejected `1`, deferred `1`
- Provider provenance visible: `15` (vertex / gemini-2.5-flash-lite)
- Evidence anchors / unknowns / uncertainty / source refs / audit reasons visible: `15` / `15` / `15` / `15` / `15`
- Hallucinated field count: `0`
- export_read_only: `True` | active_mkb_record_created_count: `0` | active_written_count: `0` | auto_accept_true_count: `0`
- review_required_true_count: `15` | live_call_made: `False` | external_api_used: `False`
- privacy_result: `passed` | billing_check_pending: `True`

## Safety

- Read-only audit/export; the 15R decision store JSONL is read and never mutated.
- Exports (JSON/CSV/markdown) are written only to the separate 15S report directory.
- No active MKB writes; no decision_status change; no auto-accept; no provider/network call; no live gate.
