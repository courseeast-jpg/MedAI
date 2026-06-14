# 17C-R2 Checkpoint and Resume Policy (R8)

## Durable checkpoint location (private, outside the repo)
`C:\Users\S1\AppData\Local\MedAI_Private\ai_extraction_17C_R2_478_live_checkpoint\`

Files (all `*_private`, never committed):

| File | Purpose |
| --- | --- |
| `checkpoint_state_private.json` | run_id, canonical batch SHA256, model, caps, total, request_index, sent/succeeded/failed counts, completed_doc_count, failed_doc_id, chunk_index/status, status |
| `checkpoint_completed_doc_ids_private.json` | list of completed document IDs/hashes (never re-sent) |
| `checkpoint_failed_doc_private.json` | failed doc ID/hash, request_index, failure_category, resolved flag |
| `checkpoint_chunk_status_private.jsonl` | per-chunk sent/succeeded/status |
| `checkpoint_provider_trace_private.jsonl` | provider **status category** per request (never a body/token) |
| `checkpoint_schema_validation_private.jsonl` | per-request schema_valid + reason |

Public reports carry only status, counts, hashes, and failure categories. Raw/private
response bodies live only in private files (and the preserved-evidence folder).

## Resume decision (`decide_resume`)
Given the canonical batch SHA256 and the in-order document IDs:

1. **No checkpoint present** → start at request #1 (`no_checkpoint_start_from_first`).
2. **State unreadable** → BLOCK (`checkpoint_state_unreadable`).
3. **Canonical batch SHA256 mismatch** → BLOCK (`canonical_batch_sha256_mismatch`).
4. **Inconsistent** (duplicate completed, unknown completed doc, or completed > total)
   → BLOCK (`checkpoint_inconsistent_*`).
5. **Unresolved failed doc** → BLOCK
   (`failed_doc_unresolved_requires_triage_or_reset`); the start index still points at the
   next unsent request, so a cleared/triaged retry resumes there.
6. **Otherwise** → resume from the first unsent request
   (`resume_from_next_unsent_request`).

## Per-request checkpointing
- On success: append the doc ID to completed, advance `request_index`, increment
  sent/succeeded, append a schema-validation line.
- On failure: write the failed-doc record, increment sent/failed, set status
  `blocked_failed_doc`, append a schema-validation line, and preserve evidence.
- Completed document IDs are **never** re-sent or re-charged on a later run.

## Operator controls
- `resolve_failed()` — explicit clearance after local triage (unblocks resume at the
  failed request).
- `reset_checkpoint()` — explicit full reset (removes all checkpoint files).

## Invariants
Stop-on-first-failure = true. MKB write = false. Auto-accept = false. Medical decision =
false. No provider/billing/model call is made by the checkpoint module.
