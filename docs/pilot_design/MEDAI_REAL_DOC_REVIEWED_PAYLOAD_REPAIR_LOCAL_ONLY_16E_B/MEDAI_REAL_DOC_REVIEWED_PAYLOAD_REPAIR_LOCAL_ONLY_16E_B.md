# MEDAI Real Document Reviewed Payload Repair Local-Only 16E-B

## Status

- 16E-B is local-only.
- 16E-B does not authorize live execution.
- 16E-B does not set the live gate.
- 16D retry is not started.
- The dedicated live gate `MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_LIVE_APPROVED` must
  remain unset/inactive throughout and after 16E-B.

## Purpose

Repair the local-only reviewed payload for the one approved real document by improving
tokenization of facility/lab identifiers and reducing false-positive tokenization of
clinical terms. The repair is based on one approved document only:
`G:\MEDICAL\Urine\Archive\2024.03.22\2.PNG`.

## Review Finding Being Repaired

- A facility/lab identifier remained raw in the 16E-A tokenized payload. Facility/lab
  identifiers such as raw `Labcorp` must be tokenized before any future live payload.
- The 16E-A tokenizer over-redacted some clinical/table terms into patient-name
  tokens, which reduced payload utility. Clinical terms and lab table labels should be
  preserved when safe.

## Repair Approach (Local Only)

- Tokenize the raw facility/lab identifier to a facility token (for example
  `[FACILITY_1]`) and add a facility/lab identifier class to the repair counts.
- Restore a curated allowlist of clinical/test terms that were incorrectly tokenized,
  so the payload keeps clinical utility while remaining de-identified.
- Keep dates tokenized unless explicitly approved later.
- Do not restore any patient identifiers.

## Privacy Model

- Raw OCR text, token maps, and the repaired tokenized payload remain private, outside
  git, under `C:\Users\S1\AppData\Local\MedAI_Private\16E_B\`.
- Public repo reports contain counts, booleans, hashes, and safe class labels only —
  no raw OCR, no token maps, and no full tokenized payload.
- Human/operator review remains mandatory before any future live retry.

## Boundaries

No Vertex call; no provider call; no Gemini/Claude/OpenAI call; no billing API call; no
live gate activation; no corpus processing; no folder processing; no additional file
processing; no MKB DB open; no active MKB write; no auto-accept; no medical decision;
no production queue mutation.

## Future Work

Future 16D retry requires explicit new authorization and is not started by this block.
