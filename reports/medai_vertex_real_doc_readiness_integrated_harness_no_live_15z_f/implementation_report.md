# MEDAI-VERTEX-REAL-DOC-READINESS-INTEGRATED-HARNESS-NO-LIVE-15Z-F

- integrated_readiness_harness_created: `True`
- integrated_cases_total: `17`
- integrated_cases_passed: `17`
- future_operator_review_package_created_count: `3`
- future_operator_review_only_count: `3`
- blocked_case_count: `14`
- refusal_records_created_count: `14`
- pii_redaction_pass_count: `15`
- vault_isolation_pass_count: `16`
- request_shape_valid_count: `15`
- review_handoff_created_count: `7`
- authorization_intent_present_count: `6`
- billing_ack_present_count: `6`
- medication_safety_proof_present_count: `3`
- explicit_live_call_request_blocked: `True`
- active_write_blocked: `True`
- auto_accept_blocked: `True`
- medication_safety_non_bypass_enforced: `True`
- medical_decision_blocked: `True`
- real_doc_live_allowed_count: `0`
- raw_pii_in_package_count: `0`
- raw_pii_in_report_count: `0`
- token_map_in_package_count: `0`
- token_map_in_report_count: `0`
- medical_decision_made_count: `0`
- live_call_made: `False`
- external_api_used: `False`
- billing_api_used: `False`
- active_written_count: `0`
- active_mkb_record_created_count: `0`
- auto_accept_true_count: `0`
- privacy_result: `passed`
- billing_check_pending: `True`

## Scope

- Composes 15Z-A through 15Z-E no-live gates into one integrated harness.
- Produces future operator review packages only; no live routing is authorized.
- Reports contain gate states, refusal reasons, and fingerprints only.
- No provider call, billing API call, active write, production queue mutation, auto-accept, or medical decision logic.
