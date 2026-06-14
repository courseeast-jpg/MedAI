# MEDAI AI-First Corpus Small Live Batch (Vertex) 17C

## Status

- 17C is the first small live AI extraction batch.
- Scope is exactly the 12 validated tokenized requests prepared by 17B-R1.
- The 587 blocked files are excluded.
- No original source files (PDF/image/OCR) are uploaded; only tokenized request text.
- No MKB write occurs in 17C. AI responses are saved to private staging only.
- Public reports contain counts and status only.

## Authorization

The user explicitly authorized this 17C small live batch using Vertex/Gemini for the 12
validated tokenized requests only, with a hard cost cap of $0.05, stop-on-first-failure,
no MKB write, no auto-accept, no medical decision, and the 17C live gate active only
inside the 17C run.

## Provider Target

- Provider: Vertex AI (Gemini), model `gemini-2.5-flash-lite`.
- No Claude, OpenAI, or other provider call. No billing API call.
- Requests are sent sequentially, one at a time, with no retry and no fallback.

## Privacy Model

- Each request is re-validated for residual raw PI patterns before any provider call.
- Raw provider responses are stored only in the private staging folder, outside the
  repo, and are never committed.
- Public reports carry counts, hashed document IDs, provider/schema status, and local
  token/cost figures only — never tokenized payloads, raw responses, token maps, raw
  OCR, or private identifier values.

## Boundaries

No raw corpus upload; no blocked-file upload; no original document upload; no MKB DB
open; no MKB write; no auto-accept; no medical decision; no production queue mutation.

## Future Work

17D MKB staging import requires separate authorization after response/schema review and
is not started by 17C.
