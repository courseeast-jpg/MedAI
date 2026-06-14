# 17C-R2 Live Entry Gate (after R8)

The runner is now checkpointed and evidence-preserving. Before any future authorized live
restart, the entry gate (computed locally, no provider/billing call) must show:

| Gate | Requirement |
| --- | --- |
| `sealed_batch_valid` | canonical 478 batch resolves and is readable |
| `credential_preflight_passed` | ADC refresh succeeds (no model call) |
| `checkpoint_clean_or_resumable` | no checkpoint, or a consistent checkpoint with matching SHA256 and no unresolved failed doc |
| `remaining_cost_estimate_within_cap` | estimated cost for the remaining (unsent) requests ≤ $0.40 total |
| `ready_to_resume_17c_r2_live` | logical AND of the above |

## Resume semantics on the next live run
- Resume continues from the next **unsent** request.
- Completed document IDs are **never** re-sent or re-charged.
- A failed document blocks resume until local triage clears it (operator-explicit
  `resolve_failed`) or the checkpoint is reset (`reset_checkpoint`).
- A canonical-batch SHA256 mismatch blocks (the batch changed under the checkpoint).
- Per-chunk cap $0.05 and total cap $0.40 still apply; stop-on-first-failure stays true.

## Current snapshot
See `reports/medai_ai_first_corpus_17c_r2_checkpointed_live_runner_hardening_local_only_17c_r2_r8/live_entry_gate_public.md`
and `summary.json` for the values produced by this block.

## After R8
If `ready_to_resume_17c_r2_live=true`: run the R4 operator-context restore, then resume
the 17C-R2 live batch once — the runner continues from the next unsent request without
re-sending completed IDs. If false: clear the reported checkpoint/batch/credential blocker
first. 17D MKB import remains not started.
