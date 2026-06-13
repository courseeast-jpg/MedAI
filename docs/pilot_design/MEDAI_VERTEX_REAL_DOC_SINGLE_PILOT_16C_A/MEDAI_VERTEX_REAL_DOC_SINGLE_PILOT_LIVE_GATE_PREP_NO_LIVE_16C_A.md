# MEDAI Vertex Real Document Single Pilot Live Gate Prep 16C-A

## Status

- This block is no-live.
- This block does not authorize live execution.
- This block does not set the live gate.
- The dedicated future live gate is only named/speced here, never set.
- This block builds on the frozen 15Z, 16A, and 16B governance state and does not
  modify it.

## Hard Boundaries

This block enforces, and the future pilot inherits: no provider call; no Vertex live
execution; no Gemini live execution; no Claude/OpenAI live execution; no billing API
call; no real/private document processing; no private corpus read; no corpus
processing; no PDF/image/OCR processing; no active MKB write; no MKB DB open; no
auto-accept; no medical decision; no production queue mutation.

## Authorization Requirements Before Any Future Live Call

- Explicit user authorization is required before any later live call.
- Separate 16D authorization is required before any one-call pilot.
- Explicit operator approval record is required before any future live run.
- Cost cap confirmation is required before any future live run.
- Redaction/tokenization preflight verification is required before any future outbound
  payload is assembled.
- One-document limit: the future pilot processes one document only.
- One-call limit: the future pilot makes exactly one call.
- Stop-on-first-failure is required in the future pilot.
- Evidence capture is required before and after any future live run.
- Rollback and failure handling are required before any future live run.

## Boundaries That Stay In Force

- No corpus processing; no whole-corpus processing.
- No active MKB write and no auto-accept during the future pilot unless separately
  authorized by a later block.
- No medical decision output.

## GCP Synthetic Sandbox Separation

The GCP synthetic sandbox evidence proves only environment, authentication, and the
request path. It does not prove MedAI real-document readiness and is separate
environment evidence only.

## 16D Status

16D is not started. No one-call pilot is initiated by this block.
