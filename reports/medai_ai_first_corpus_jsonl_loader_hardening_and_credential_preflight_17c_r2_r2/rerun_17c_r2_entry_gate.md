# 17C-R2 rerun entry gate (after loader hardening + preflight)

- sealed_batch_valid: `True`
- credential_preflight_passed: `True` (adc: `pass`)
- ready_to_rerun_17c_r2_live: `True`

17C-R2 is NOT re-run here. Re-run only when the sealed batch is valid AND the
credential preflight passes, keeping the run chunked/capped with
stop-on-first-failure and the live gate scoped to each chunk.
