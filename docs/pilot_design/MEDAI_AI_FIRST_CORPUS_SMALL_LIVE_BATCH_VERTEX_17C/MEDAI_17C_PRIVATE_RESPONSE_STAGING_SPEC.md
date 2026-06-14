# MEDAI 17C Private Response Staging Spec

All raw provider responses and tokenized request bodies are kept private, outside the
repo, under `C:\Users\S1\AppData\Local\MedAI_Private\ai_extraction_17C_live_batch\`.

Private staging files (never committed):

- live_request_manifest_private.json
- live_responses_private.jsonl
- parsed_responses_private.jsonl
- schema_validation_private.json
- provider_trace_private.json
- cost_guard_private.json
- stopped_on_failure_private.json (if applicable)

Public repo reports contain counts, hashed document IDs, provider/schema status, and
local token/cost figures only. No tokenized payloads, raw AI responses, token maps,
raw OCR, or private identifier values appear in the repo or in Downloads.
