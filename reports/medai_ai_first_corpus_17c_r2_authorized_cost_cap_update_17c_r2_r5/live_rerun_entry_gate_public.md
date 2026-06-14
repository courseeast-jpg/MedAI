# 17C-R2 live rerun entry gate (after R5 cap update)

- authorized_hard_cost_cap_total_usd: `0.4` (was `0.25`)
- hard_cost_cap_per_chunk_usd: `0.05` | chunk_size: `25`
- latest_estimated_total_cost_usd: `0.325362` | within_new_cap: `True`
- sealed_batch_valid: `True` | credential_preflight_passed: `True` (adc: `pass`)
- ready_to_rerun_17c_r2_live: `True`

17C-R2 is NOT re-run here. Recommended order: run the R4 operator-context restore
immediately before the live run (the private batch can be deleted between blocks),
then run 17C-R2 once. The runner re-verifies the batch at preflight and enforces
the $0.05/chunk and $0.40 total caps with stop-on-first-failure.
