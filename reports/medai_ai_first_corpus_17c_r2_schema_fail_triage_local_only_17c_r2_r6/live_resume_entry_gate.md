# 17C-R2 live resume entry gate (after R6)

- ready_to_resume_17c_r2_live: `True`
- resume_policy: `restart_required`
- sealed_batch_valid: `True` | credential_preflight_passed: `True`

The 17C-R2 runner has no per-request checkpoint, so a resume re-sends from the
first request (request #1 is sent again). If the failure recurs at request #2,
capture the private response body and re-triage; the normalizer intentionally does
not recover prose/truncated/safety-refusal responses.
