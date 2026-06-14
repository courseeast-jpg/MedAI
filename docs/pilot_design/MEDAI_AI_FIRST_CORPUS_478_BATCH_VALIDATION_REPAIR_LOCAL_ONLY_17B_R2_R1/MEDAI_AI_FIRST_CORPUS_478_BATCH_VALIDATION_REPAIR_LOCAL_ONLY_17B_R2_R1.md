# MEDAI AI-First Corpus 478 Batch Validation Repair Local-Only 17B-R2-R1

## Status

- 17B-R2-R1 is local-only.
- 17B-R2-R1 does not call any AI provider (no Vertex, Gemini, Claude, or OpenAI call).
- 17B-R2-R1 does not start live extraction.
- No live gate is set. MKB import remains not started.

## Purpose

Repair the 42 residual identifier-pattern validation failures from 17B-R2 and rebuild a
clean 478-file private outbound request batch, then re-validate under the unchanged
17B-R2 rules.

## Scope

- The 478 ready files are the total ready set. The old 12-file batch is included in the
  478 total (proven by doc_id dedupe); the old 12 must not be added again.
- No 490-file batch is created.
- Only the 42 failed documents are repaired; all 478 are rebuilt as outbound requests.
- Duplicates, unsupported files, remaining extraction_unavailable files, and all
  non-ready files are excluded. No original source files are processed.

## Privacy Model

- Tokenized content is read locally from the 17A tokenized corpus; outbound requests are
  built privately, outside the repo, and never committed.
- Public reports carry counts, hashed document IDs, token/cost estimates, and validation
  status only — never tokenized payloads, raw OCR, token maps, residual numeric values,
  or private identifier values.

## Conservative Tokenization

Identifier tokenization is intentionally conservative: any residual sequence that
triggers a validator class is tokenized before any future AI upload, even when the value
might be a clinically useful barcode/account/specimen identifier. Privacy outranks
preserving such identifiers for the first live corpus pass.

## Future Work

Future live extraction requires separate authorization and working Vertex credentials
(see the credential preflight). It is not started here.
