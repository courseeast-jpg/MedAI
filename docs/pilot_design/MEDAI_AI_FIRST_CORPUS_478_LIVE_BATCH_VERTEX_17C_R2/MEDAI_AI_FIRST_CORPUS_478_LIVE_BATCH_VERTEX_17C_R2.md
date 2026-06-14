# MEDAI AI-First Corpus 478 Live Batch (Vertex) 17C-R2

## Status

- 17C-R2 is a live Vertex/Gemini extraction run.
- Scope is exactly the canonical 478 validated tokenized requests from 17B-R2-R1.
- The old 12-file batch is not separate and must not be added; combined batch count is 478.
- No original source files (PDF/image/OCR) are uploaded — only tokenized request text.
- No MKB write occurs. AI responses are saved to private staging only.
- Public reports contain counts and status only.
- Non-ready files (blocked, duplicate, extraction_unavailable, unsupported) remain excluded.

## Authorization

The user authorized 17C-R2 live extraction for the canonical 478 validated tokenized
requests only, using gemini-2.5-flash-lite, chunk size 25, hard cost cap $0.05 per chunk
and $0.25 total, stop-on-first-failure, no MKB write, no auto-accept, no medical
decision, and the 17C-R2 live gate active only inside each authorized chunk.

## Guardrails

- Strict preflight: the canonical batch must load as exactly 478 well-formed requests;
  any count mismatch or malformed line blocks the run before any provider call.
- Privacy re-validation of every request before any provider call.
- Local token/cost estimate with hard per-chunk and total caps; no billing API call.
- Credential preflight (ADC refresh for project sot-knowledge-ocr) before any model call.
- Sequential chunks and sequential requests; no retry, no fallback, no parallel calls.

## Future Work

17D MKB staging import requires separate authorization after response/schema review and
is not started by 17C-R2.
