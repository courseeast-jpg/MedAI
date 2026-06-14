# MEDAI-AI-FIRST-CORPUS-17C-R2-MISSING-SCHEMA-FIELDS-TRIAGE-LOCAL-ONLY-17C-R2-R7

## Result: **PASS** (safety: passed)

## Metrics

- prior_live_execution_result: `SCHEMA_FAIL`
- prior_request_count_loaded: `478`
- prior_request_count_sent: `2`
- prior_request_count_succeeded: `1`
- prior_request_count_failed: `1`
- prior_failure_stage: `schema`
- prior_failure_category: `missing_expected_schema_fields`
- failed_request_index: `2`
- failed_doc_id_public: `doc_015d3c111db4712d`
- failed_response_body_available_in_context: `False`
- schema_failure_class: `unknown_missing_expected_schema_fields`
- missing_required_field_count: `0`
- missing_required_field_names: `[]`
- provider_returned_json_object: `True`
- response_privacy_clean: `False`
- prompt_hardened_for_required_fields: `True`
- schema_skeleton_instruction_added: `True`
- required_fields_precheck_added: `True`
- schema_weakened: `False`
- required_fields_made_optional: `False`
- clinical_values_inferred: `False`
- missing_required_values_synthesized: `False`
- local_replay_recovered_failed_response: `False`
- local_replay_schema_valid_count: `1`
- local_replay_schema_failed_count: `1`
- authorized_hard_cost_cap_total_usd: `0.4`
- hard_cost_cap_per_chunk_usd: `0.05`
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

- run the R4 operator-context restore, then restart 17C-R2 live once per the resume policy.
