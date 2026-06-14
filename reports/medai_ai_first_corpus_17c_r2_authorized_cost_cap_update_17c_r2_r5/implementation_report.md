# MEDAI-AI-FIRST-CORPUS-17C-R2-AUTHORIZED-COST-CAP-UPDATE-17C-R2-R5

## Result: **PASS** (safety: passed)

## Metrics

- target_model: `gemini-2.5-flash-lite`
- chunk_size: `25`
- hard_cost_cap_per_chunk_usd: `0.05`
- previous_hard_cost_cap_total_usd: `0.25`
- authorized_hard_cost_cap_total_usd: `0.4`
- runner_total_cap_matches_authorized: `True`
- caps_consistent_in_runner: `True`
- latest_estimated_total_cost_usd: `0.325362`
- estimated_total_within_new_cap: `True`
- request_count_loaded: `478`
- sealed_batch_valid: `True`
- credential_preflight_passed: `True`
- project_detected: `sot-knowledge-ocr`
- adc_refresh_result: `pass`
- ready_to_rerun_17c_r2_live: `True`
- provider_model_call_made: `False`
- vertex_model_call_made: `False`
- billing_api_call_made: `False`
- live_gate_set: `False`
- live_extraction_started: `False`
- mkb_db_opened: `False`
- active_mkb_write: `False`
- private_outbound_requests_committed: `False`
- public_report_phi_leak_count: `0`
- privacy_result: `passed`
- safety_result: `passed`

## Recommended next (no live run started)

- run the R4 operator-context restore, then rerun 17C-R2 live extraction once.
