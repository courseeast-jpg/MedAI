# MEDAI-VERTEX-SEMANTIC-REVIEW-DECISION-PANEL-TAB-WIRE-15U

- Nav entry registered + dispatch wired: `True`
- Nav label: `Vertex Decision Audit` | reachable in app: `True`
- Core tabs preserved (Run & Review / MKB Explorer / Review Queue): `True`
- Advanced tab labels: `['Run & Review', 'MKB Explorer', 'Review Queue', 'Operator Control Panel', 'Validation Batch Audit', 'Validation History', 'Safety & Governance', 'Terminology Admin', 'Vertex Decision Audit']`
- Panel rendered from nav: `True` | decisions visible: `15`
- Decision summary: total `15`, accepted_for_review `13`, rejected `1`, deferred `1`
- Package families visible: `{'cytology_pathology_narrative': 3, 'mixed_narrative_numeric_result': 4, 'portal_result_cards': 4, 'urinalysis_table_like_lab': 4}`
- Provider route/model visible: `15` / `15`
- Evidence/source/ref/audit visible: `15` / `15` / `15` / `15`
- Unknown/uncertainty visible: `15` / `15`
- Hallucinated field count visible: `0`
- Export affordances (JSON/CSV/MD): `True` / `True` / `True`
- decision_store_unchanged: `True` | active_mkb_record_created_count: `0` | active_written_count: `0` | auto_accept_true_count: `0`
- live_call_made: `False` | external_api_used: `False` | privacy_result: `passed`

## Safety

- Additive nav registration only: the new advanced tab routes to the existing 15T read-only hook.
- Core operator tabs (Run & Review / MKB Explorer / Review Queue) are unchanged and still first.
- The 15R decision store is read only and never mutated (fingerprint verified). No provider call; no live gate.
