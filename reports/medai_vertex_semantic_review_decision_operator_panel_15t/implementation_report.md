# MEDAI-VERTEX-SEMANTIC-REVIEW-DECISION-OPERATOR-PANEL-15T

- Panel rendered: `True` | decisions visible: `15`
- Decision summary: total `15`, accepted_for_review `13`, rejected `1`, deferred `1`
- Package families visible: `{'cytology_pathology_narrative': 3, 'mixed_narrative_numeric_result': 4, 'portal_result_cards': 4, 'urinalysis_table_like_lab': 4}`
- Provider route/model visible: `15` / `15` (vertex / gemini-2.5-flash-lite)
- Evidence anchor / source evidence / source ref / audit reason visible: `15` / `15` / `15` / `15`
- Unknown / uncertainty visible: `15` / `15`
- Hallucinated field count visible: `0`
- Export affordances visible (JSON/CSV/MD): `True` / `True` / `True`
- export_read_only: `True` | decision_store_unchanged: `True`
- active_mkb_record_created_count: `0` | active_written_count: `0` | auto_accept_true_count: `0`
- live_call_made: `False` | external_api_used: `False` | privacy_result: `passed` | billing_check_pending: `True`

## Safety

- Compact read-only operator panel over the 15S audit/export view-model.
- The 15R decision store is read only and never mutated (fingerprint verified before/after).
- Export affordances reference already-generated 15S artifacts; no active MKB writes; no decision_status change.
- No auto-accept; no provider/network call; no live gate. Additive UI hook only (no broad redesign).
