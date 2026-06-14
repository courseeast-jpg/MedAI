# MEDAI-AI-FIRST-CORPUS-17C-R2-CHECKPOINTED-LIVE-RUNNER-HARDENING-LOCAL-ONLY-17C-R2-R8 — implementation report

## What changed
- Added `execution/live_checkpoint.py`: durable per-request checkpoint, strict resume policy, and failed-evidence preservation — all OUTSIDE the repo.
- Patched the 17C-R2 live runner to resume from the next unsent request, skip completed documents, checkpoint after every request, block on SHA256 mismatch / unresolved failure / inconsistency, and immediately preserve failed evidence out of the volatile staging folder.

## Checkpoint files (private)
- `checkpoint_state_private.json`
- `checkpoint_completed_doc_ids_private.json`
- `checkpoint_failed_doc_private.json`
- `checkpoint_chunk_status_private.jsonl`
- `checkpoint_provider_trace_private.jsonl`
- `checkpoint_schema_validation_private.jsonl`

## Resume policy
- resume_from_next_unsent_request: completed doc IDs are never re-sent / re-charged.
- Blocks on: canonical batch SHA256 mismatch, unresolved failed doc (until local triage / operator reset), inconsistent or unreadable checkpoint.
- No checkpoint present -> start from request #1. Stop-on-first-failure remains true.

## Simulation (provider-free, isolated temp dirs)
- request #1 schema-valid: `True` -> completed.
- request #2 schema-fail (`missing_expected_schema_fields`) -> failed + evidence preserved.
- resume next index (1-based): `2` (does NOT re-send request #1: resends_completed_request=`False`).
- failed doc blocks until triaged: `True`; unblocks after resolve: `True`.
- SHA256 mismatch blocks: `True`.
- provider call made: `False`; live gate set: `False`.

## Live entry gate
- ready_to_resume_17c_r2_live: `True` (sealed_batch_valid=`True`, credentials=`True`, remaining_within_cap=`True`).

## Safety
- No provider/billing/model call; no live gate; no MKB; no private response bodies committed.
