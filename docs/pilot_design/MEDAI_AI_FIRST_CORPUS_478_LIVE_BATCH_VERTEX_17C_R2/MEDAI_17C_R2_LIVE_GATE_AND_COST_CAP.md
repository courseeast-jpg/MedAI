# MEDAI 17C-R2 Live Gate And Cost Cap

## Live Gate

- 17C-R2 uses a dedicated gate: `MEDAI_AI_FIRST_CORPUS_478_LIVE_BATCH_17C_R2_APPROVED=YES`.
- The gate must be inactive before the script starts.
- The script activates the gate only inside live chunk execution and clears it after every
  chunk and in a final finally block.
- The public report confirms the final gate state is inactive.
- The old 16D gate and the old 17C 12-file gate are not used.

## Cost Cap

- Hard cost cap: $0.05 per chunk and $0.25 total for this run.
- Token/cost is estimated locally before any provider call; no billing API call is made.
- If the total estimate exceeds $0.25, or any chunk estimate exceeds $0.05, the run is
  blocked before any provider call.

## Batch

- Exactly the canonical 478 validated tokenized requests. Chunk size 25 (20 chunks).
- No 490-file batch; the old 12 are already included in the 478.
