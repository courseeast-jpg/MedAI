# MEDAI-AI-FIRST-CORPUS-17C-R2-CHECKPOINTED-LIVE-RUNNER-HARDENING-LOCAL-ONLY-17C-R2-R8

Local-only hardening. **No** provider/Gemini/Vertex/Claude/OpenAI call, **no** billing
call, **no** live gate, **no** MKB open/write, **no** corpus request. This block converts
the 17C-R2 live runner from a restart-only prototype into a checkpointed,
evidence-preserving batch runner and proves the new behavior with a provider-free,
isolated simulation.

## Why
The 17C-R2 live route reached Vertex/Gemini twice. Both attempts stopped safely after 2
sent requests (succeeded 1, failed 1). R6 hardened strict-JSON parsing; R7 required all
core schema keys. Three operational blockers remained:

1. **No per-request checkpoint** — `restart_required` re-sent and re-charged request #1 on
   every retry.
2. **Volatile evidence** — an external writer cleared the `MedAI_Private` live-staging
   folder, destroying the failed-response body before triage (twice).
3. **High operator time** — every failure forced manual recovery.

## What this block delivers
- `execution/live_checkpoint.py` — durable, outside-the-repo checkpoint state; a strict
  resume policy; and immediate failed-evidence preservation.
- A minimal patch to `scripts/run_medai_ai_first_corpus_478_live_batch_vertex_17c_r2.py`
  to consult/update the checkpoint per request and preserve evidence on failure.
- A provider-free simulation (isolated temp dirs) proving: success #1 → resume at #2,
  request #1 never re-sent, failed evidence preserved, live gate never set.
- A post-hardening live entry gate.

## Boundaries (unchanged, still enforced)
Stop-on-first-failure remains true. MKB write, auto-accept, and medical decision remain
false. The only approved real pilot input remains `G:\MEDICAL\Urine\Archive\2024.03.22\2.PNG`.
The 478 canonical tokenized requests are the only authorized live scope; non-ready files
stay excluded. Durable checkpoint state and preserved evidence are PRIVATE and are never
committed.

## Companion documents
- `MEDAI_17C_R2_CHECKPOINT_AND_RESUME_POLICY_R8.md`
- `MEDAI_17C_R2_FAILED_EVIDENCE_PRESERVATION_POLICY_R8.md`
- `MEDAI_17C_R2_LIVE_ENTRY_GATE_AFTER_R8.md`
