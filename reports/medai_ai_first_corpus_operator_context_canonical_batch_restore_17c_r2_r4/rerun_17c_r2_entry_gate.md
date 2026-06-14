# 17C-R2 live rerun entry gate (after R4)

- canonical_batch_exists_after: `True`
- canonical_batch_readable_after: `True` (readonly: `True`)
- sealed_batch_valid: `True`
- credential_preflight_passed: `True` (adc: `pass`)
- ready_to_rerun_17c_r2_live: `True`

17C-R2 is NOT re-run here. The live runner must re-verify the canonical batch
(existence + SHA256 + counts) immediately before any provider call.
