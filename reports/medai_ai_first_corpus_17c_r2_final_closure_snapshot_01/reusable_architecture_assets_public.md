# Corpus 1 / 17C-R2 — Reusable Architecture Assets

These local-only assets were built across the 17C-R2 program and are reusable for Corpus 2
(and any future corpus) without modification. None of them perform a provider call by
themselves.

| Asset | Origin | Reusable for |
| --- | --- | --- |
| Strict-JSON response normalizer (`execution/strict_json.py`) | R6 | rejecting non-strict / prose / multiple / truncated responses |
| Required top-level key precheck + prompt skeleton | R7 | enforcing the full schema without weakening it |
| Durable checkpoint + resume (`execution/live_checkpoint.py`) | R8 | resume-from-next-unsent; never re-send completed doc IDs |
| Failed-evidence preservation | R8 | copying failed bodies out of volatile staging before triage |
| Public-report redaction (`execution/public_report_redaction.py`) | R9 | path/secret labels so public reports pass the privacy checker |
| Raised output ceiling (8192) + compact-output prompt rules | R10 | preventing MAX_TOKENS mid-JSON truncation |
| Adaptive cost + chunk planner (`execution/cost_chunk_planner.py`) | R10 | keeping each chunk within the per-chunk cap |
| Authorized total cap update ($10.00) | R11 | full-corpus budget gating |
| Checkpoint unblock + redaction hardening | R12 | clearing a stale failed-doc block under operator control |
| Sectioned extraction + autonomous recovery runner | R13 | per-section requests + self-resuming full-corpus recovery |
| Provider hard-stop diagnosis | R14 | classifying provider-side `api_disabled_or_permission` failures |
| Resume supervisor + live report | R15 | supervised resume reporting |

## Carry-over for Corpus 2
- The privacy gate, tokenization discipline, strict-JSON contract, checkpoint/resume, and
  redaction helpers apply unchanged.
- The Corpus 1 failure was provider/cloud side (`api_disabled_or_permission`), not an
  extraction-code defect, so the architecture itself is sound to reuse.
