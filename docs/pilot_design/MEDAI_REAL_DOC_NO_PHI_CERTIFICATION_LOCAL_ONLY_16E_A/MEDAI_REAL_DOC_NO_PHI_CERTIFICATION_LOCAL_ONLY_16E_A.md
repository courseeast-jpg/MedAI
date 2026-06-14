# MEDAI Real Document No-PHI Certification Local-Only 16E-A

## Status

- 16E-A is local-only.
- 16E-A does not authorize live execution.
- 16E-A does not set the live gate.
- 16D retry is not started.
- The dedicated live gate `MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_LIVE_APPROVED` must
  remain unset/inactive throughout and after 16E-A.

## Purpose

Perform a local-only no-PHI certification assessment for the single approved image
that previously blocked 16D. The goal is to determine, without any provider call,
whether a future outbound payload could ever be reviewed and certified free of raw
patient identifiers before a later, separately authorized live attempt.

## Approved Input

- One document only: `G:\MEDICAL\Urine\Archive\2024.03.22\2.PNG`.
- No other file, no folder, no corpus is processed.

## Privacy Model

- Raw artifacts (raw OCR text, raw extracted text, token maps, tokenized payloads,
  operator-review material) stay private and outside git, under
  `C:\Users\S1\AppData\Local\MedAI_Private\16E_A\`.
- Public reports contain counts, hashes, pass/fail flags, and sanitized summaries
  only. No raw OCR text, no token maps, and no PHI/PII appear in repo artifacts.

## Outcome Semantics

- This block never auto-certifies a real document as safe to send.
- The honest outcome for an unstructured real scanned document is
  `NEEDS_HUMAN_REVIEW`: an automated detector cannot prove the absence of every
  identifier.
- Human/operator review is required before any later live send.

## Boundaries

No Vertex call; no provider call; no Gemini/Claude/OpenAI call; no billing API call;
no live gate activation; no corpus processing; no folder processing; no additional
file processing; no MKB DB open; no active MKB write; no auto-accept; no medical
decision; no production queue mutation.

## Future Work

Future 16D retry requires explicit new authorization and is not started by this block.
