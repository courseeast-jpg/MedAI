# MEDAI-VERTEX-REAL-DOC-ADAPTER-DRY-RUN-NO-LIVE-15Z-C

- adapter_dry_run_created: `True`
- adapter_cases_total: `14`
- adapter_cases_passed: `14`
- request_shape_valid_count: `5`
- request_top_level_keys_exact_count: `5`
- forbidden_metadata_rejected_count: `1`
- generation_config_valid_count: `5`
- would_be_request_fingerprints_created_count: `14`
- outbound_payload_fingerprints_created_count: `14`
- vault_fingerprints_referenced_count: `14`
- review_queue_handoff_records_created_count: `5`
- review_queue_handoff_report_only_count: `5`
- readiness_cases_fed_count: `14`
- no_live_replay_allowed_count: `4`
- blocked_case_count: `9`
- future_authorization_only_count: `1`
- real_doc_live_allowed_count: `0`
- raw_pii_in_request_count: `0`
- raw_pii_in_report_count: `0`
- token_map_in_request_count: `0`
- token_map_in_report_count: `0`
- live_call_made: `False`
- external_api_used: `False`
- active_written_count: `0`
- active_mkb_record_created_count: `0`
- auto_accept_true_count: `0`
- privacy_result: `passed`
- billing_check_pending: `True`

## Scope

- Builds the would-be Vertex request shape offline from 15Z-B tokenized payloads.
- Keeps provider metadata local to dry-run proof and review handoff records.
- Writes handoff records only to this report folder; no active MKB or production queue write occurs.
- Keeps `live_call_allowed=false`, `active_write_allowed=false`, `auto_accept_allowed=false`, and `review_required=true`.
