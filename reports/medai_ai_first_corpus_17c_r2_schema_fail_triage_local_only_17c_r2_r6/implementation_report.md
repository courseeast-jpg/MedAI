# MEDAI-AI-FIRST-CORPUS-17C-R2-SCHEMA-FAIL-TRIAGE-LOCAL-ONLY-17C-R2-R6

## Result: **PASS** (safety: passed)

## Metrics

- prior_live_execution_result: `SCHEMA_FAIL`
- prior_request_count_loaded: `478`
- prior_request_count_sent: `2`
- prior_request_count_succeeded: `1`
- prior_request_count_failed: `1`
- prior_failure_stage: `schema`
- prior_failure_category: `not_strict_json`
- failed_request_index: `2`
- failed_doc_id_public: `doc_015d3c111db4712d`
- failed_response_body_available_in_context: `False`
- schema_failure_class: `unknown_not_strict_json`
- safe_normalizer_added: `True`
- normalizer_synthetic_replay_all_pass: `True`
- prompt_hardened_for_strict_json: `True`
- json_mode_enabled_if_supported: `True`
- schema_weakened: `False`
- partial_json_accepted: `False`
- missing_fields_inferred: `False`
- local_replay_attempted: `True`
- local_replay_recovered_failed_response: `False`
- local_replay_schema_valid_count: `1`
- local_replay_schema_failed_count: `1`
- authorized_hard_cost_cap_total_usd: `0.4`
- hard_cost_cap_per_chunk_usd: `0.05`
- stale_cap_test_updated: `True`
- sealed_batch_valid: `True`
- credential_preflight_passed: `True`
- ready_to_resume_17c_r2_live: `True`
- resume_policy: `restart_required`
- provider_model_call_made: `False`
- live_gate_set: `False`
- live_extraction_started: `False`
- mkb_db_opened: `False`
- active_mkb_write: `False`
- public_report_phi_leak_count: `0`
- privacy_result: `passed`
- safety_result: `passed`

## Recommended next (no live run started)

- run the R4 operator-context restore, then resume/restart 17C-R2 live once per the resume policy.
