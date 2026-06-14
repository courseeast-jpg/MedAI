# 17C-R2 live resume entry gate (after R7)

- ready_to_resume_17c_r2_live: `True`
- resume_policy: `restart_required`
- sealed_batch_valid: `True` | credential_preflight_passed: `True`

The 17C-R2 runner has no per-request checkpoint, so a resume re-sends from the
first request. The exact missing fields for the prior failure were unavailable;
watch request #2 on the next run and capture the body if it fails again.
