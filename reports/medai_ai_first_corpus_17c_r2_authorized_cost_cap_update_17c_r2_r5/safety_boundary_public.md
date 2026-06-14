# 17C-R2-R5 safety boundary

- No Gemini/Vertex/Claude/OpenAI model call; no provider content request.
- No billing API call; cost caps enforced from local estimates only.
- No live gate activation; no live extraction; no corpus request sent.
- No MKB DB open/write; no auto-accept; no medical decision; no queue mutation.
- Only the total cap constant changed (0.25 -> 0.40). Per-chunk cap (0.05),
  chunk size (25), model (gemini-2.5-flash-lite), privacy validation, response
  schema, and the prompt contract are unchanged.
- No private outbound bodies, raw OCR, token maps, PI values, or credentials
  are printed or committed.
