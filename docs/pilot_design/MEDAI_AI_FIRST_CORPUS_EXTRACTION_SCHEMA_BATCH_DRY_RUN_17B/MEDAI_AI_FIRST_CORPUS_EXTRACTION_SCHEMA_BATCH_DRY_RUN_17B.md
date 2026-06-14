# MEDAI AI-First Corpus Extraction Schema Batch Dry-Run 17B

## Status

- 17B is dry-run only.
- 17B does not call any AI provider (no Vertex, Gemini, Claude, or OpenAI call).
- 17B does not upload corpus data.
- 17B uses only the 12 ready tokenized files marked `ready_for_ai_extraction` in 17A.
- 17B prepares the later 17C small live batch; it does not start it.
- The dedicated live gate `MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_LIVE_APPROVED` remains
  unset/inactive throughout and after 17B.

## Purpose

Build the AI-first extraction stage artifacts for the 12 ready tokenized corpus files:
the extraction JSON schema, the prompt contract, the batch manifest, the outbound
request payload builder, the response validator, and a local cost estimator — all as a
no-provider dry run.

## Scope

- Only the 12 files marked `ready_for_ai_extraction` from the private 17A tokenized
  corpus package are used.
- The 587 blocked files are explicitly excluded from AI extraction until fixed.
- This block does not repair the `extraction_unavailable` blockers.
- This block does not start live AI extraction.

## Privacy Model

- Private tokenized text is read locally for the 12 ready documents only. Outbound
  request JSONL is built privately, outside the repo.
- Public repo reports contain only counts, schema names, hashed document IDs, token
  counts, estimated tokens/cost, and validation status.
- Tokenized payloads, raw OCR, token maps, and private PI vault values are never
  committed.

## Technical Debt Note

Temporary technical debt is accepted for UI and normalization work, not for privacy.
Privacy gates are never deferred.

## Boundaries

No provider call; no billing API call; no live gate activation; no MKB DB open; no
active MKB write; no auto-accept; no medical decision; no production queue mutation.

## Future Work

The 12 ready files feed a later 17C small live batch, which requires explicit new
authorization and is not started here.
