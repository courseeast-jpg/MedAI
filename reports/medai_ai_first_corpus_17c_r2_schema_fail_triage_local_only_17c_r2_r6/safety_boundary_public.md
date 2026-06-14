# 17C-R2-R6 safety boundary

- No Gemini/Vertex/Claude/OpenAI model call; no provider content request.
- No billing API call; no live gate; no live extraction; no corpus request.
- No MKB open/write; no auto-accept; no medical decision; no queue mutation.
- No raw AI response bodies, parsed responses, tokenized payloads, raw OCR, token
  maps, private values, or credentials are printed or committed.
- Schema not weakened; partial JSON not accepted; missing fields not inferred.
