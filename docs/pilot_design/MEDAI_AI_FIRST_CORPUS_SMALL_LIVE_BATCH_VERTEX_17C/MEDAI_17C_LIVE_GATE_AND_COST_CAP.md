# MEDAI 17C Live Gate And Cost Cap

## Live Gate

- 17C uses a dedicated 17C-specific live gate: `MEDAI_AI_FIRST_CORPUS_SMALL_LIVE_BATCH_17C_APPROVED=YES`.
- The gate must be inactive before the script starts.
- The script activates the gate only inside the live execution process.
- The script deactivates/clears the gate before exit, including all error paths
  (finally block).
- The public report confirms the final gate state is inactive.
- The older single-document 16D gate (`MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_LIVE_APPROVED`)
  and the smoke gate (`MEDAI_VERTEX_LIVE_SMOKE_ALLOWED`) are NOT set or relied upon.

## Hard Cost Cap

- Hard cost cap for the 17C run: $0.05 (maximum estimated cost).
- Before any provider call, the script estimates total input/output tokens and the
  approximate cost using local token approximation and configurable per-million prices.
- If the estimate exceeds $0.05, the run is blocked before any provider call.
- No billing API call is made; the cap is enforced from local estimates.

## Batch Limit

- Exactly 12 requests maximum. The script refuses to run if the loaded request count is
  not 12. No additional request and no blocked file is processed.
