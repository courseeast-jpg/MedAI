# MEDAI-AI-FIRST-CORPUS-JSONL-LOADER-HARDENING-AND-CREDENTIAL-PREFLIGHT-17C-R2-R2

## Result: **PASS** (safety: passed)

## Metrics

- jsonl_splitlines_usage_removed_from_live_loader: `True`
- jsonl_physical_newline_reader_used: `True`
- sha256_sidecar_verified: `True`
- sealed_batch_valid: `True`
- request_count_loaded: `478`
- physical_newline_record_count: `478`
- parseable_record_count: `478`
- unique_doc_ids: `478`
- malformed_record_count: `0`
- request_validation_passed: `True`
- residual_pi_failures: `0`
- old_12_added_separately: `False`
- combined_batch_count: `478`
- credential_preflight_attempted: `True`
- credential_preflight_passed: `True`
- project_detected: `sot-knowledge-ocr`
- adc_refresh_result: `pass`
- ready_to_rerun_17c_r2_live: `True`
- provider_model_call_made: `False`
- vertex_model_call_made: `False`
- gemini_call_made: `False`
- billing_api_call_made: `False`
- live_gate_set: `False`
- live_extraction_started: `False`
- mkb_db_opened: `False`
- active_mkb_write: `False`
- private_outbound_requests_committed: `False`
- tokenized_payloads_written_to_repo: `False`
- raw_ocr_written_to_repo: `False`
- token_maps_written_to_repo: `False`
- private_identifier_values_written_to_repo: `False`
- credential_or_token_written_to_repo: `False`
- public_report_phi_leak_count: `0`
- privacy_result: `passed`
- safety_result: `passed`

## Recommended next (no live run started)

- rerun 17C-R2 live extraction once.
