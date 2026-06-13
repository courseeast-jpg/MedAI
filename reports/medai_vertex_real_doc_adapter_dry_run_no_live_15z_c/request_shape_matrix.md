# 15Z-C request shape matrix

| Case | Adapter status | Shape valid | Request keys | Handoff | Readiness | Blocked |
| --- | --- | --- | --- | --- | --- | --- |
| sanitized_redacted_real_like_basic_note | DRY_RUN_REQUEST_SHAPE_VALID_REVIEW_REQUIRED | True | contents, generationConfig | True | READY_FOR_NO_LIVE_REPLAY_ONLY | False |
| sanitized_redacted_real_like_lab_report | DRY_RUN_REQUEST_SHAPE_VALID_REVIEW_REQUIRED | True | contents, generationConfig | True | READY_FOR_NO_LIVE_REPLAY_ONLY | False |
| sanitized_contact_fields_fixture | DRY_RUN_REQUEST_SHAPE_VALID_REVIEW_REQUIRED | True | contents, generationConfig | True | READY_FOR_NO_LIVE_REPLAY_ONLY | False |
| all_9_pii_classes_sanitized_fixture | DRY_RUN_REQUEST_SHAPE_VALID_REVIEW_REQUIRED | True | contents, generationConfig | True | READY_FOR_NO_LIVE_REPLAY_ONLY | False |
| unknown_provenance_payload | BLOCKED | True | contents, generationConfig | False | BLOCKED_ADAPTER_DRY_RUN | True |
| real_private_marker_payload | BLOCKED | True | contents, generationConfig | False | BLOCKED_ADAPTER_DRY_RUN | True |
| payload_with_raw_pii_residue | BLOCKED | True | contents, generationConfig | False | BLOCKED_ADAPTER_DRY_RUN | True |
| payload_with_token_map_leak | BLOCKED | True | contents, generationConfig | False | BLOCKED_ADAPTER_DRY_RUN | True |
| request_with_forbidden_metadata_top_level_key | BLOCKED | False | contents, generationConfig, metadata | False | BLOCKED_ADAPTER_DRY_RUN | True |
| request_with_invalid_generation_config | BLOCKED | False | contents, generationConfig | False | BLOCKED_ADAPTER_DRY_RUN | True |
| active_write_requested | BLOCKED | True | contents, generationConfig | False | BLOCKED_ADAPTER_DRY_RUN | True |
| auto_accept_requested | BLOCKED | True | contents, generationConfig | False | BLOCKED_ADAPTER_DRY_RUN | True |
| medication_fact_without_safety_gate | BLOCKED | True | contents, generationConfig | False | BLOCKED_ADAPTER_DRY_RUN | True |
| all_future_gates_simulated_pass | READY_FOR_FUTURE_AUTHORIZATION_ONLY | True | contents, generationConfig | True | READY_FOR_FUTURE_AUTHORIZATION_ONLY | False |

| Metric | Value |
| --- | --- |
| adapter_cases_total | `14` |
| request_shape_valid_count | `5` |
| request_top_level_keys_exact_count | `5` |
| forbidden_metadata_rejected_count | `1` |
| generation_config_valid_count | `5` |
| review_queue_handoff_records_created_count | `5` |
| blocked_case_count | `9` |
| future_authorization_only_count | `1` |
| real_doc_live_allowed_count | `0` |
