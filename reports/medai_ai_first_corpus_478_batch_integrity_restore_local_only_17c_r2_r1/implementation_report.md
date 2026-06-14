# MEDAI-AI-FIRST-CORPUS-478-BATCH-INTEGRITY-RESTORE-LOCAL-ONLY-17C-R2-R1

## Result: **PASS** (safety: passed)

## Metrics

- bad_lines_before: `591`
- bad_parseable_before: `457`
- bad_malformed_before: `134`
- expected_request_count: `478`
- rebuilt_request_count: `478`
- parseable_after: `478`
- unique_doc_ids_after: `478`
- malformed_after: `0`
- request_validation_passed_after: `True`
- residual_pi_failures_after: `0`
- old_12_added_separately: `False`
- combined_batch_count: `478`
- private_integrity_sidecar_written: `True`
- sha256_sidecar_written: `True`
- doc_id_manifest_written: `True`
- file_set_readonly_after_restore: `True`
- provider_call_made: `False`
- gemini_call_made: `False`
- claude_call_made: `False`
- openai_call_made: `False`
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
- public_report_phi_leak_count: `0`
- future_17c_r2_live_not_started: `True`
- privacy_result: `passed`
- safety_result: `passed`

## Recommended next (no live run started)

- rerun the Vertex credential preflight; if credentials PASS, rerun 17C-R2 live extraction.
