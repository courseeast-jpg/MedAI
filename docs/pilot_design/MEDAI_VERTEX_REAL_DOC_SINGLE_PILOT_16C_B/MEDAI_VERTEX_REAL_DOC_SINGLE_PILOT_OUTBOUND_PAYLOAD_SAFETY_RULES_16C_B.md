# MEDAI Vertex Real Document Single Pilot Outbound Payload Safety Rules 16C-B

## Status

No-live, synthetic only. No provider call, no Vertex live execution, no billing API
call. 16D is not started.

## Outbound Payload Rules

- The outbound payload must contain tokens only; no raw identifier may appear.
- The outbound payload must pass the redaction/tokenization preflight before assembly.
- Token maps must never be included in outbound payload.
- Any raw identifier in the outbound payload is a hard NO-GO and triggers
  stop-on-first-failure.
- Future request shape is limited to tokenized content; no provider/MedAI metadata is
  attached.

## Boundaries

No real/private document processing; no private corpus read; no corpus processing; no
PDF/image/OCR processing; no MKB write; no auto-accept; no medical decision; no
production queue mutation.
