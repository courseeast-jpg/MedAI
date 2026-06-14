# MEDAI AI-First Corpus 17B Phone-Pattern Repair Local-Only 17B-R1

## Status

- 17B-R1 is local-only.
- 17B-R1 does not call any AI provider (no Vertex, Gemini, Claude, or OpenAI call).
- 17B-R1 does not start 17C.
- The dedicated live gate `MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_LIVE_APPROVED` remains
  unset/inactive throughout and after 17B-R1.

## Purpose

Repair the 4 tokenized corpus payloads that failed 17B dry-run validation due to
residual phone-like / 10-digit-style numeric sequences, then rebuild the 12-file
private outbound request batch and re-validate, with the goal of
`request_validation_passed=true`.

## Approach

- Read the private 17B validator output to identify the failed document IDs and confirm
  the failure class is phone-pattern only.
- Read only the 12 ready tokenized payloads; repair only the 4 failed documents.
- Replace every flagged 10-digit-style sequence with a token; never preserve a flagged
  raw sequence in the outbound payload.
- Rebuild the 12-request outbound JSONL privately, outside the repo, and re-validate.

## Conservative Repair

The repair is intentionally conservative: any possible phone-like numeric sequence that
triggered validation is tokenized, even when it might be a benign barcode/specimen/
order/accession number. Fail-closed safety is preferred over payload utility.

## Scope

- The 587 blocked files remain excluded from AI extraction.
- This block does not repair the `extraction_unavailable` blockers.
- A future 17C live batch requires explicit new authorization and is not started here.

## Privacy Model

- Private tokenized/repaired payloads and outbound requests are written outside the
  repo and never committed.
- Public repo reports contain counts, hashed document IDs, token counts, estimates, and
  validation status only — never tokenized payloads, raw OCR, token maps, PI values, or
  the flagged numeric values themselves.

## Boundaries

No provider call; no billing API call; no live gate activation; no MKB DB open; no
active MKB write; no auto-accept; no medical decision; no production queue mutation.
