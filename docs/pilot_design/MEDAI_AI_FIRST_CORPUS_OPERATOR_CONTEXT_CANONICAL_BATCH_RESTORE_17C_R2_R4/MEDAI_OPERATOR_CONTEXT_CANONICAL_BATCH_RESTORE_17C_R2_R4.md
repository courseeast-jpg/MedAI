# MEDAI Operator-Context Canonical Batch Restore 17C-R2-R4

## Status

- Local/preflight only. No Gemini/Vertex/Claude/OpenAI model call, no provider content
  request, no billing API call, no live gate, no live extraction.
- No MKB open/write, no auto-accept, no medical decision, no production queue mutation.

## Problem

The 17C-R2 live runner and the R3 resolver tests intermittently reported
`canonical_batch_missing` in the operator runtime even though prior reports claimed the
sealed batch existed. The operator-context check is the source of truth: a repo report
is not proof that the private file is present on disk in the current runtime. An external
writer in this worktree has been deleting/mutating files under `MedAI_Private`, so the
private canonical batch can disappear between blocks.

## What This Block Does

1. Diagnoses the canonical path in the live operator context (presence, is_file,
   readable, read-only, sidecars) — public-safe booleans/paths only, no file bodies.
2. If the canonical batch or its sidecars are missing or invalid, re-materializes them
   from the 17A tokenized corpus using the 17B-R2-R1 repair logic: writes
   `outbound_requests_private.jsonl` with `ensure_ascii=True` and physical-newline
   framing, plus the integrity JSON, SHA256, and doc-id manifest sidecars, then sets the
   batch read-only.
3. Verifies the sealed batch (SHA256 + 478/478/478/0 + 0 residual PI) and confirms the
   R3 resolver tests pass in the same context.
4. Runs the Vertex credential preflight (ADC refresh, no model call, no token printed).

If the private tokenized corpus is also missing, the block stops with BLOCKED and
reports `private_tokenized_corpus_missing` — it never fabricates requests.

## Privacy

Private outbound bodies, raw OCR, token maps, PI values, and credentials are never
printed or committed. Public reports carry booleans, counts, hashes, and the resolved
path string only.
