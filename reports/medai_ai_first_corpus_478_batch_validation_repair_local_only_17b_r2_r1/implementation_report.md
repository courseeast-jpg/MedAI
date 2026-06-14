# MEDAI-AI-FIRST-CORPUS-478-BATCH-VALIDATION-REPAIR-LOCAL-ONLY-17B-R2-R1

## Result: **PASS** (safety: passed)

## Metrics

- ready_files_total: `478`
- combined_batch_count: `478`
- old_12_added_separately: `False`
- doc_id_dedupe_check_performed: `True`
- old_12_included_in_478: `True`
- failed_files_from_17b_r2: `42`
- failure_classes_from_17b_r2: `{'accession_specimen': 18, 'insurance_account': 15, 'phone': 15, 'mrn': 2}`
- files_repaired: `42`
- outbound_requests_built: `478`
- request_validation_passed: `True`
- request_validation_failed_count_after_repair: `0`
- validation_failure_classes_after_repair: `{}`
- estimated_input_tokens: `422381`
- estimated_output_tokens: `382400`
- estimated_cost_usd: `0.146399`
- suggested_live_batch_size: `25`
- suggested_live_batch_count: `20`
- suggested_hard_cost_cap_usd: `0.05`
- provider_call_made: `False`
- gemini_call_made: `False`
- claude_call_made: `False`
- openai_call_made: `False`
- billing_api_call_made: `False`
- live_gate_set: `False`
- live_extraction_started: `False`
- raw_source_files_uploaded: `False`
- tokenized_payloads_written_to_repo: `False`
- raw_ocr_written_to_repo: `False`
- token_maps_written_to_repo: `False`
- private_identifier_values_written_to_repo: `False`
- public_report_phi_leak_count: `0`
- mkb_db_opened: `False`
- active_mkb_write: `False`
- auto_accept_enabled: `False`
- medical_decision_made: `False`
- production_queue_mutated: `False`
- future_live_extraction_not_started: `True`
- privacy_result: `passed`
- safety_result: `passed`

## Recommended next (no live extraction started)

- prepare 478-file live extraction in bounded chunks after confirming Vertex credential preflight PASS.
