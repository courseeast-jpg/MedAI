# MEDAI-AI-FIRST-CORPUS-EXTRACTION-SCHEMA-BATCH-DRY-RUN-17B

## Result: **BLOCKED** (safety: passed)

## Result

- Dry-run preparation of AI-first extraction for the 12 ready tokenized files.
- Built private outbound request JSONL (outside the repo) and validated each
  request for residual raw PI. Estimated tokens/cost locally.
- No provider call, no network, no billing, no live gate activation.

## Metrics

- dry_run_only: `True`
- ready_files_from_17a: `12`
- blocked_files_excluded: `587`
- outbound_requests_built: `8`
- request_validation_passed: `False`
- validation_failures: `4`
- estimated_input_tokens: `5010`
- estimated_output_tokens: `6400`
- estimated_cost_usd: `0.002296`
- target_model_for_future_live: `gemini-2.5-flash-lite`
- schema_created: `True`
- prompt_contract_created: `True`
- private_payloads_written_outside_repo: `True`
- tokenized_payloads_written_to_repo: `False`
- raw_ocr_written_to_repo: `False`
- token_maps_written_to_repo: `False`
- private_identifier_values_written_to_repo: `False`
- public_report_phi_leak_count: `0`
- provider_call_made: `False`
- gemini_call_made: `False`
- claude_call_made: `False`
- openai_call_made: `False`
- billing_api_call_made: `False`
- future_live_gate_environment_active: `False`
- mkb_db_opened: `False`
- active_mkb_write: `False`
- auto_accept_enabled: `False`
- medical_decision_made: `False`
- production_queue_mutated: `False`
- future_17c_live_not_started: `True`
- privacy_result: `blocked`
- safety_result: `passed`

## Recommended next (NO live batch started automatically)

- fix failed payload validation before any live batch. 17C requires explicit new authorization and is not started.
