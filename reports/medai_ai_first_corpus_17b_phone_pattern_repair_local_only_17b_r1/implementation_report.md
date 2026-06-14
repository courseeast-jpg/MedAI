# MEDAI-AI-FIRST-CORPUS-17B-PHONE-PATTERN-REPAIR-LOCAL-ONLY-17B-R1

## Result: **PASS** (safety: passed)

## Result

- Repaired the 4 phone-pattern failures by tokenizing flagged numeric sequences,
  then rebuilt and re-validated the 12-request private outbound batch.
- Validator rules unchanged; payloads repaired first. No provider call, no network,
  no billing, no live gate activation.

## Metrics

- ready_files_from_17a: `12`
- failed_files_from_17b: `4`
- failed_reason: `phone_pattern`
- files_repaired: `4`
- outbound_requests_built: `12`
- request_validation_passed: `True`
- residual_phone_pattern_failures_after_repair: `0`
- estimated_input_tokens: `41036`
- estimated_output_tokens: `9600`
- estimated_cost_usd: `0.005958`
- target_model_for_future_live: `gemini-2.5-flash-lite`
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
- seventeen_c_live_started: `False`
- mkb_db_opened: `False`
- active_mkb_write: `False`
- auto_accept_enabled: `False`
- medical_decision_made: `False`
- production_queue_mutated: `False`
- blocked_files_excluded: `587`
- future_17c_live_not_started: `True`
- privacy_result: `passed`
- safety_result: `passed`

## Recommended next (NO live batch started automatically)

- prepare 17C small live batch authorization for the 12 validated requests. 17C requires explicit new authorization and is not started.
