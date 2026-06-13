# MEDAI Vertex Real Document Single Pilot 16D Handoff Template 16C-C

This template prepares, but does not start, 16D. 16D is not started. This block is
no-live and does not set the live gate.

## Handoff Package Contents (To Be Completed Before Any Authorized 16D)

- Final operator approval record (signed).
- Explicit user authorization reference.
- Cost cap confirmation reference.
- Redaction/tokenization proof reference (no raw PII/PHI; no token map in outbound).
- One-document and one-call limit confirmation.
- Stop-on-first-failure plan.
- Rollback/failure plan.
- Evidence capture plan (before and after).
- Dedicated live gate name: `MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_LIVE_APPROVED` (to be set only inside the
  authorized 16D block; unset/inactive now).

## Stop Conditions Carried Into 16D

16D must stop if the live gate, approval record, cost cap, redaction proof, or
one-call limit is missing. No provider call, no corpus processing, no PDF/image/OCR
processing, no MKB write, no auto-accept, and no medical decision occur outside an
explicitly authorized 16D execution.
