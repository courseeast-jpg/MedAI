# 15Z-D authorization cost refusal matrix

| Case | Auth intent | Billing ack | Handoff | Future package | Status | Blocked |
| --- | --- | --- | --- | --- | --- | --- |
| missing_human_authorization | False | True | True | False | BLOCKED | True |
| missing_billing_ack | True | False | True | False | BLOCKED | True |
| missing_both_authorization_and_billing | False | False | True | False | BLOCKED | True |
| valid_no_live_authorization_intent_only | True | False | True | False | BLOCKED | True |
| valid_no_live_billing_ack_only | False | True | True | False | BLOCKED | True |
| authorization_and_billing_present_but_no_review_handoff | False | False | False | False | BLOCKED | True |
| authorization_and_billing_present_with_15z_c_handoff | True | True | True | True | READY_FOR_FUTURE_AUTHORIZATION_PACKAGE_ONLY | False |
| real_private_provenance | True | True | True | False | BLOCKED | True |
| unknown_provenance | True | True | True | False | BLOCKED | True |
| active_write_requested | True | True | True | False | BLOCKED | True |
| auto_accept_requested | True | True | True | False | BLOCKED | True |
| medication_fact_without_safety_gate | True | True | True | False | BLOCKED | True |
| forbidden_request_metadata | True | True | True | False | BLOCKED | True |
| invalid_generation_config | True | True | True | False | BLOCKED | True |
| raw_pii_detected | True | True | True | False | BLOCKED | True |
| token_map_leak_detected | True | True | True | False | BLOCKED | True |
| explicit_live_call_requested | True | True | True | False | BLOCKED | True |

| Metric | Value |
| --- | --- |
| authorization_cases_total | `17` |
| blocked_case_count | `16` |
| refusal_records_created_count | `16` |
| future_authorization_package_created_count | `1` |
| real_doc_live_allowed_count | `0` |
| no_billing_api_used | `True` |
