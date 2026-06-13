# MEDAI-VERTEX-REAL-DOC-AUTHORIZATION-COST-REFUSAL-GATES-NO-LIVE-15Z-D

- authorization_cost_refusal_gates_created: `True`
- authorization_cases_total: `17`
- authorization_cases_passed: `17`
- blocked_case_count: `16`
- refusal_records_created_count: `16`
- human_authorization_required_count: `4`
- billing_ack_required_count: `4`
- authorization_intent_created_count: `13`
- billing_ack_created_count: `13`
- future_authorization_package_created_count: `1`
- future_authorization_only_count: `1`
- real_doc_live_allowed_count: `0`
- explicit_live_call_request_blocked: `True`
- active_write_blocked: `True`
- auto_accept_blocked: `True`
- medication_safety_non_bypass_enforced: `True`
- no_billing_api_used: `True`
- estimated_cost_ceiling_usd_max: `0.01`
- token_budget_ceiling_max: `512`
- raw_pii_in_report_count: `0`
- token_map_in_report_count: `0`
- live_call_made: `False`
- external_api_used: `False`
- active_written_count: `0`
- active_mkb_record_created_count: `0`
- auto_accept_true_count: `0`
- privacy_result: `passed`
- billing_check_pending: `True`

## Scope

- Models no-live human authorization intent and billing/cost-cap acknowledgement.
- Creates sanitized refusal records for every blocked path.
- Creates a future authorization package preview only; no live call is authorized.
- No provider call, no billing API call, no active MKB write, no production queue mutation, no auto-accept.
