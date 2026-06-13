# 15Z-F integrated gate trace matrix

| Case | Status | Package | PII | Vault | Request shape | Handoff | Auth | Billing | Med proof | Blocked |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean_redacted_real_like_non_medication_full_path | READY_FOR_FUTURE_OPERATOR_REVIEW_PACKAGE_ONLY | True | True | True | True | True | True | True | False | False |
| clean_redacted_real_like_medication_full_path_with_safety_proof | READY_FOR_FUTURE_OPERATOR_REVIEW_PACKAGE_ONLY | True | True | True | True | True | True | True | True | False |
| unknown_provenance | BLOCKED | False | True | True | True | False | False | False | False | True |
| real_private_marker | BLOCKED | False | True | True | True | False | False | False | False | True |
| raw_pii_residue | BLOCKED | False | False | True | True | False | False | False | False | True |
| token_map_leak | BLOCKED | False | False | False | True | False | False | False | False | True |
| forbidden_request_metadata | BLOCKED | False | True | True | False | False | False | False | False | True |
| invalid_generation_config | BLOCKED | False | True | True | False | False | False | False | False | True |
| missing_review_handoff | BLOCKED | False | True | True | True | False | False | False | False | True |
| missing_human_authorization | BLOCKED | False | True | True | True | True | False | True | False | True |
| missing_billing_ack | BLOCKED | False | True | True | True | True | True | False | False | True |
| active_write_requested | BLOCKED | False | True | True | True | False | False | False | False | True |
| auto_accept_requested | BLOCKED | False | True | True | True | False | False | False | False | True |
| medication_without_safety_proof | BLOCKED | False | True | True | True | False | False | False | False | True |
| medication_decision_requested | BLOCKED | False | True | True | True | True | True | True | True | True |
| explicit_live_call_requested | BLOCKED | False | True | True | True | True | True | True | False | True |
| all_gates_simulated_pass_still_no_live | READY_FOR_FUTURE_OPERATOR_REVIEW_PACKAGE_ONLY | True | True | True | True | True | True | True | True | False |

| Metric | Value |
| --- | --- |
| integrated_cases_total | `17` |
| future_operator_review_package_created_count | `3` |
| blocked_case_count | `14` |
| real_doc_live_allowed_count | `0` |
