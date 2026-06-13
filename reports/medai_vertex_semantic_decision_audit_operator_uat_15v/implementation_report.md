# MEDAI-VERTEX-SEMANTIC-DECISION-AUDIT-OPERATOR-UAT-15V

- UAT passed: `True`
- Operator nav entry found / tab opened / panel rendered: `True` / `True` / `True`
- Panel title / read-only indicator / read-only statement visible: `True` / `True` / `True`
- Decisions loaded/visible: `15` / `15` (accepted_for_review `13`, rejected `1`, deferred `1`)
- Package family breakdown correct: `True` — `{'cytology_pathology_narrative': 3, 'mixed_narrative_numeric_result': 4, 'portal_result_cards': 4, 'urinalysis_table_like_lab': 4}`
- Provider route/model visible: `15` / `15`
- Evidence/source/ref/audit visible: `15` / `15` / `15` / `15`
- Unknown/uncertainty visible: `15` / `15`
- Hallucinated field count visible: `0`
- Export affordances visible (JSON/CSV/MD): `True` / `True` / `True`
- Export artifacts exist (JSON/CSV/MD): `True` / `True` / `True`
- decision_store_unchanged: `True` | active_mkb_record_created_count: `0` | active_written_count: `0` | auto_accept_true_count: `0`
- live_call_made: `False` | external_api_used: `False` | privacy_result: `passed`

## Safety

- Bounded deterministic UAT over the reachable 'Vertex Decision Audit' tab via the app dispatch path and 15T view-model.
- No browser automation, no provider/network call, no live gate, no active MKB write.
- 15R decision store read only; fingerprint verified unchanged before/after the UAT journey.
