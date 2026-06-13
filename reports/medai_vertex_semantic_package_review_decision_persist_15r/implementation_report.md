# MEDAI-VERTEX-SEMANTIC-PACKAGE-REVIEW-DECISION-PERSIST-15R

- Decision store (isolated, local-only): `reports/medai_vertex_semantic_package_review_decision_persist_15r/decision_store_preview.jsonl`
- Package families loaded / with decisions: `4` / `4`
- Decision records created: `15` (accept `13`, reject `1`, defer `1`)
- Invalid actions rejected (no record written): `1`
- Review-bound decision count: `15`
- Active MKB records created: `0` | Active written count: `0`
- Auto-accept true count: `0` | Review required true count: `15`
- Evidence anchors / provider provenance preserved: `15` / `15`
- Unknown / uncertainty preserved: `15` / `15`
- Hallucinated field count: `0`
- live_call_made: `False` | external_api_used: `False`
- privacy_result: `passed` | billing_check_pending: `True`

## Safety

- Decisions persist to an isolated JSONL review-draft store; no production MKB/ledger/queue write path is touched.
- Accept = 'accepted for review queue only'; Reject = 'rejected - no active write'; Defer = 'deferred - no active write'.
- creates_active_mkb_record=false and active_written_count_delta=0 on every record; auto_accept stays false; review_required stays true.
- Evidence anchors and Vertex provider provenance preserved on every record. No provider call; no live gate.
