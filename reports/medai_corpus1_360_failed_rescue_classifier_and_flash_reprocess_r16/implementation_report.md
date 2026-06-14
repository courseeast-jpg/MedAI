# MEDAI-CORPUS1-360-FAILED-RESCUE-CLASSIFIER-AND-FLASH-REPROCESS-R16 — implementation report

- Rescue model: `gemini-2.5-flash` (was `gemini-2.5-flash-lite`).
- Selected 360 failed_for_review docs; 118 completed preserved/excluded.
- Failure taxonomy: {'api_permission_or_route': 360, 'provider_safety_finish': 0, 'provider_max_tokens': 0, 'provider_empty_response': 0, 'provider_invalid_json': 0, 'schema_validation_failed': 0, 'section_merge_failed': 0, 'content_too_large': 0, 'retry_exhausted': 0, 'unknown_provider_hard_stop': 0, 'evidence_missing': 0, 'checkpoint_inconsistent': 0}.
- Estimated cost `$7.417904` within remaining `$49.971853` of the shared $50 pool.
- live_entry_gate_passed: `True`; run_result: `LIVE_FAIL`.
- completed_after: `0`; failed_after: `360`.
- Sectioned/autonomous recovery; separate R16 checkpoint; main checkpoint untouched; no MKB.
