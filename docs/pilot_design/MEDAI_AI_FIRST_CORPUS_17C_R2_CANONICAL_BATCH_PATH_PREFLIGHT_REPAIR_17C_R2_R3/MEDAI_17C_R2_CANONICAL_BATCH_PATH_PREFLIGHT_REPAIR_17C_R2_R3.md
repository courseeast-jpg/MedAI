# MEDAI 17C-R2 Canonical Batch Path Preflight Repair 17C-R2-R3

## Status

- Local/preflight only. No Gemini/Vertex/Claude/OpenAI model call, no provider content
  request, no billing API call, no live gate, no live extraction.
- No MKB open/write, no auto-accept, no medical decision, no production queue mutation.

## Root Cause

The 17C-R2 live runner resolved the canonical private batch path only via
`os.path.expandvars("%LOCALAPPDATA%\...")`. When `LOCALAPPDATA` is unset/empty in the
runtime environment, `expandvars` returns the unexpanded literal `%LOCALAPPDATA%\...`,
which is not a real path, so `is_file()` is False and the preflight falsely reports
`canonical_batch_missing` with `request_count_loaded=0`. The file itself was present,
sealed, and valid (478) the whole time — this was a path-resolution defect, not missing
data, and not the read-only attribute (read-only files still resolve as files).

## Fix

A shared resolver (`execution/canonical_batch_paths.py`) resolves the canonical batch
by trying, in order: the explicit expected path
`C:\Users\S1\AppData\Local\MedAI_Private\ai_extraction_17B_R2_R1_478_repaired\outbound_requests_private.jsonl`,
the `%LOCALAPPDATA%` expansion, and a `Path.home()`-based path; the first that exists
wins. The 17C-R2 live runner now uses this resolver (path resolution only; live
execution behavior, cost cap, chunk size, stop-on-first-failure, and safety gates are
unchanged). The sealed batch is loaded with the physical-newline reader from
`execution/jsonl_framing.py`.

## Verification

This block re-verifies path resolution, sidecars, SHA256, the 478/478/478/0 sealed
batch (physical-newline framing), 0 residual PI, and the Vertex credential preflight
(no model call, no token printed).
