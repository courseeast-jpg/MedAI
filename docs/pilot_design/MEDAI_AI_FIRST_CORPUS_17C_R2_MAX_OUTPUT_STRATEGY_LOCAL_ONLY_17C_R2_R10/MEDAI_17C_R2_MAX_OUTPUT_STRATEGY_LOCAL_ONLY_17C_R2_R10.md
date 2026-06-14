# MEDAI-AI-FIRST-CORPUS-17C-R2-MAX-OUTPUT-STRATEGY-LOCAL-ONLY-17C-R2-R10

Local-only strategy fix for the confirmed `provider_truncated_by_max_tokens` blocker.
**No** provider/Gemini/Vertex/Claude/OpenAI call, **no** billing call, **no** live gate,
**no** MKB open/write, **no** corpus request. No response body, tokenized payload, token
map, private value, or credential printed or committed.

## Root cause (from R9)
The latest live run hit the 2048 output-token ceiling (`finishReason=MAX_TOKENS`,
`candidatesTokenCount=2048`) and the JSON object was cut off mid-structure.

## Changes
1. **Output ceiling raised 2048 -> 8192** in the live runner. gemini-2.5-flash-lite
   supports at least 8192 output tokens; the adapter posts `maxOutputTokens` verbatim and
   imposes no lower ceiling.
2. **Compact-output prompt rules** added to the contract (see
   `MEDAI_17C_R2_COMPACT_JSON_OUTPUT_POLICY_R10.md`) so responses stay small while still
   emitting one strict JSON object with every required top-level key. Schema is **not**
   weakened.
3. **Adaptive chunk-size planning** (`execution/cost_chunk_planner.py`) recomputes cost at
   the new ceiling and selects the largest chunk size that keeps each chunk within the
   $0.05 per-chunk cap. Total request count is unchanged.
4. **Cost guards recalculated** at the new ceiling; hard caps preserved ($0.40 total,
   $0.05 per chunk).
5. **R8 checkpoint/resume and failed-evidence preservation preserved.**
6. **R9 public-report redaction preserved.**

## Cost finding (worst case at the 8192 ceiling)
- Per-chunk: adaptive chunk size **17** (from 25) keeps each chunk within $0.05.
- Total: the worst-case total for all 478 requests at 8192 is **above** the $0.40 cap.

## Consequence
The per-chunk cap is satisfiable, but the worst-case total exceeds $0.40. Therefore
`ready_to_resume_17c_r2_live=false` and `requires_new_cost_authorization=true`: a revised
total-cost authorization, or a separately authorized smaller live batch, is needed before
any live run at this ceiling. The total request count is unchanged by design, so the only
levers are a higher cap or a smaller authorized batch.

See `MEDAI_17C_R2_LIVE_ENTRY_GATE_AFTER_R10.md` for the full gate.
