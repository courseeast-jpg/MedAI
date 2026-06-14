# MEDAI-AI-FIRST-CORPUS-17C-R2-CANONICAL-BATCH-PATH-PREFLIGHT-REPAIR-17C-R2-R3

## Result: **PASS** (safety: passed)

## Root cause

- The 17C-R2 runner resolved the canonical path only via `%LOCALAPPDATA%` expansion;
  when that variable is unset, `expandvars` returns a literal nonexistent path and the
  preflight falsely reported `canonical_batch_missing`. Fixed with a shared resolver
  (explicit path -> %LOCALAPPDATA% -> Path.home()).

## Metrics

- canonical_batch_resolved_path: `C:\Users\S1\AppData\Local\MedAI_Private\ai_extraction_17B_R2_R1_478_repaired\outbound_requests_private.jsonl`
- canonical_batch_exists: `True`
- canonical_batch_is_file: `True`
- canonical_batch_readable: `True`
- canonical_batch_readonly: `True`
- canonical_batch_parent_exists: `True`
- canonical_batch_missing_reason: ``
- integrity_sidecar_exists: `True`
- sha256_sidecar_exists: `True`
- doc_id_manifest_exists: `True`
- sha256_sidecar_verified: `True`
- sealed_batch_valid: `True`
- request_count_loaded: `478`
- physical_newline_record_count: `478`
- parseable_record_count: `478`
- unique_doc_ids: `478`
- malformed_record_count: `0`
- request_validation_passed: `True`
- residual_pi_failures: `0`
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

- rerun 17C-R2 live extraction once.
