# 17C-R2 Failed-Evidence Preservation Policy (R8)

## Problem
The live-staging folder
`C:\Users\S1\AppData\Local\MedAI_Private\ai_extraction_17C_R2_478_live_batch\` is volatile:
an external writer has cleared it before triage, destroying the failed-response body
twice. R8 copies failure evidence out of that folder the moment a failure occurs.

## Preservation location (private, outside the repo, never committed)
`C:\Users\S1\Downloads\MedAI_17C_R2_FAILED_EVIDENCE_PRESERVE_PRIVATE\run_<timestamp>\`

A unique `run_<timestamp>` subfolder is created on each failure.

## What is preserved (private copies)
- failed raw provider response (`live_responses_private.jsonl`)
- parsed response if available (`parsed_responses_private.jsonl`)
- failed doc metadata (`schema_validation_private.json`)
- schema validation detail + provider trace (`provider_trace_private.json`)
- stop reason (`stopped_on_failure_private.json`)
- copies of the checkpoint state files
- `README_PRIVATE_DO_NOT_SHARE.txt` — sanitized warning: may contain raw/parsed bodies
  and tokenized fragments; do not commit, paste, email, or upload.

Preservation is best-effort: the run folder is always created, and only sources that
still exist are copied (`files_copied` records how much survived). Nothing is fabricated.

## Public reporting
Public reports carry **only**:
- `evidence_preserved` (true/false)
- `preservation_path`
- failed doc **hash**
- failure **category**

No raw body, no parsed body, no tokenized payload, no token map, no credential ever
appears in a public report or is committed to the repo.

## Boundaries
The preservation folder is PRIVATE and is excluded from all commits. Preservation itself
makes no provider/billing/model call and sets no live gate.
