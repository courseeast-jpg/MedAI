# 17C-R2 Compact JSON Output Policy (R10)

The output ceiling was raised 2048 -> 8192 to stop mid-JSON truncation. To keep responses
small (and within cost caps) without relaxing the schema, the prompt contract now requires
compact output.

## Compact rules (added to `config/medai_ai_extraction_prompt_contract_17b.md`)
- JSON object only — no explanatory text, preamble, reasoning, or commentary.
- No duplicate evidence: the same evidence anchor is never repeated across fields.
- No long copied passages: large spans of the document are not pasted.
- Evidence anchors must be short — the minimal tokenized snippet that supports a fact.
- Optional verbose notes are omitted unless the schema requires them; no narrative fields
  beyond those the schema defines.
- Empty arrays `[]` for absent sections; every required top-level key still present.
- Arrays stay compact: one concise object per finding, no padding or restated context.

## What is preserved (not weakened)
- One JSON object only; no markdown fences; no prose (R6 strict JSON).
- All required top-level keys present, even when empty (R7 skeleton).
- Local strict-JSON normalization and required-key precheck still reject incomplete or
  non-strict output.

The compact rules reduce response size; they do not change which fields are required or
allow partial/inferred output.
