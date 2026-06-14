# MEDAI Real Document Repaired Payload Machine Review Local-Only 16E-C

## Status

- 16E-C is local-only.
- 16E-C does not authorize live execution.
- 16E-C does not set the live gate.
- 16D retry is not started.
- The dedicated live gate `MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_LIVE_APPROVED` must
  remain unset/inactive throughout and after 16E-C.

## Purpose

Perform a local, automated machine review of the repaired tokenized payload produced
by 16E-B for the single approved document. The review produces sanitized counts and
pass/fail flags only. It does not constitute human attestation and never auto-marks
NO_PHI_ATTESTED.

## Inputs (Local Only)

- `repaired_tokenized_payload.txt` (private review copy)
- `repair_review.json` (private review copy)
- Optional local inspection of the one approved source image.

## Review Checks

The machine review confirms, at heuristic confidence only:

1. Raw `Labcorp` is absent.
2. No obvious raw patient name remains.
3. No DOB remains.
4. No address remains.
5. No phone/email remains.
6. No MRN remains.
7. No insurance/account/accession/specimen ID remains.
8. No provider/facility/lab identifier remains except tokens such as `[FACILITY_1]`.
9. No local file path appears in the payload.
10. Remaining `[PATIENT_NAME_*]` tokens do not expose raw identifiers.
11. Clinical table content remains usable enough for extraction review.
12. If uncertain, the result is NEEDS_HUMAN_REVIEW, not PASS.

## Outcome Semantics

- Machine review is heuristic and cannot prove the absence of every identifier.
- The result is therefore PASS only when no targeted identifier pattern is found AND
  confidence is high; otherwise NEEDS_HUMAN_REVIEW.
- This block never writes NO_PHI_ATTESTED and never starts 16D.

## Privacy Model

- Raw OCR, token maps, and the repaired payload are never committed.
- Public reports contain counts, booleans, and hashes only — no raw payload body, no
  token map, no raw OCR.

## Boundaries

No Vertex/provider/Gemini/Claude/OpenAI call; no billing API call; no live gate
activation; no MKB DB open; no active MKB write; no auto-accept; no medical decision;
no production queue mutation.

## Future Work

Operator attestation remains required. A future 16D retry requires explicit new
authorization and is not started by this block.
