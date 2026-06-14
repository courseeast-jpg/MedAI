# MEDAI 17C-R2 Private Response Staging Spec

All raw provider responses, parsed outputs, and tokenized request bodies are kept
private, outside the repo, under
`C:\Users\S1\AppData\Local\MedAI_Private\ai_extraction_17C_R2_478_live_batch\`.

Private staging files (never committed):

- live_request_manifest_private.json
- live_chunk_plan_private.json
- live_responses_private.jsonl
- parsed_responses_private.jsonl
- schema_validation_private.json
- provider_trace_private.json
- cost_guard_private.json
- stopped_on_failure_private.json (if applicable)
- chunk_status_private.jsonl
- completed_doc_ids_private.json
- failed_doc_private.json (if applicable)

Public repo reports contain counts, chunk numbers, hashed document IDs, provider/schema
status, and local token/cost figures only. No tokenized payloads, raw AI responses,
parsed response bodies, token maps, raw OCR, private identifier values, or credentials
appear in the repo or in Downloads.
