# 17C-R2 live entry gate (after R8)

- sealed_batch_valid: `True`
- credential_preflight_passed: `True` (`pass`)
- checkpoint_clean_or_resumable: `True` (`no_checkpoint_start_from_first`)
- remaining_requests: `478`
- remaining_cost_estimate_usd: `0.338017`
- authorized_hard_cost_cap_total_usd: `0.4`
- hard_cost_cap_per_chunk_usd: `0.05`
- remaining_cost_estimate_within_cap: `True`
- resume_policy: `resume_from_next_unsent_request`
- ready_to_resume_17c_r2_live: `True`

Resume continues from the next unsent request; completed document IDs are never re-sent or re-charged. A failed document blocks resume until local triage clears it (operator-explicit resolve) or the checkpoint is reset.
