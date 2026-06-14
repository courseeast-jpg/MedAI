# 17C-R2 live rerun entry gate (after R3)

- canonical_batch_resolved_path: `C:\Users\S1\AppData\Local\MedAI_Private\ai_extraction_17B_R2_R1_478_repaired\outbound_requests_private.jsonl`
- canonical_batch_is_file / readable: `True` / `True`
- sealed_batch_valid: `True`
- credential_preflight_passed: `True` (adc: `pass`)
- ready_to_rerun_17c_r2_live: `True`

17C-R2 is NOT re-run here. Re-run only when the canonical batch resolves, the
sealed batch is valid, and the credential preflight passes — keeping the run
chunked/capped with stop-on-first-failure and the live gate scoped to each chunk.
