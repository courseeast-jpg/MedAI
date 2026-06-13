# MEDAI-VERTEX-SEMANTIC-CALIBRATION-BATCH-SYNTHETIC-LIVE-15X

- Status: `PASS`
- Fixtures / live calls / responses: `15` / `15` / `15`
- Schema pass / hallucinated: `15` / `0`
- Source body / evidence / candidate-sep / unknown / uncertainty preserved: `15` / `15` / `15` / `15` / `15`
- Allowed-keys-only / review-required: `15` / `15`
- Tokens (prompt/output/total): `4950` / `2384` / `7334`
- Estimated cost all calls / ceiling (USD): `0.0014486` / `0.00512`
- Cost constants: input $0.0001/1K, output $0.0004/1K tokens.
- auto_accept_true_count: `0` | active_written_count: `0` | active_mkb_record_created_count: `0`
- live_call_made: `True` | external_api_used: `True` | privacy_result: `passed` | billing_check_pending: `True`
- stopped_early: `False` | stop_reason: ``

## Safety

- Synthetic/redacted fixtures only; <=20 live calls (one per fixture); no retries; stop on first failure.
- Every posted body contains only `contents` + `generationConfig`; no MedAI metadata sent.
- No active MKB writes; output review-bound; no auto-accept; no provider call without the 15X gate.
- Token/cost ledger uses documented conservative local constants; billing_check_pending=true.
