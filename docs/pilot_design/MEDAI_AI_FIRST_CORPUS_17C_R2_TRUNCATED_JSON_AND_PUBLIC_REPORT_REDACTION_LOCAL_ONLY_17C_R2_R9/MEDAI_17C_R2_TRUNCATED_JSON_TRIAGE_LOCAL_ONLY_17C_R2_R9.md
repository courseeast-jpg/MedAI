# MEDAI-AI-FIRST-CORPUS-17C-R2-TRUNCATED-JSON-AND-PUBLIC-REPORT-REDACTION-LOCAL-ONLY-17C-R2-R9

Local-only triage + redaction repair. **No** provider/Gemini/Vertex/Claude/OpenAI call,
**no** billing call, **no** live gate, **no** MKB open/write, **no** corpus request. No raw
response body, tokenized payload, token map, private value, or credential was printed or
committed.

## The failure
The latest 17C-R2 live run reached Vertex/Gemini and stopped on **request #1**:

- execution_result `SCHEMA_FAIL`, sent 1, succeeded 0, failed 1, stage `schema`,
  category `truncated_or_invalid_json`.

## Classification (from PRIVATE preserved evidence, structural metadata only)
The R8 evidence-preservation copied the failed body out of the volatile staging folder
before the external writer cleared it, so the body was available privately for triage.

Structural metadata (no body emitted):
- `finishReason = MAX_TOKENS`
- `candidatesTokenCount = 2048` — exactly the per-request output ceiling
- `starts_with_object = true`, `ends_with_object = false`, `brace_balance > 0`
- JSON parse error: "Invalid control character" (a secondary symptom of the cut-off tail)

→ **`truncated_json_failure_class = provider_truncated_by_max_tokens`**.

The model's JSON output exceeded the per-request output-token ceiling and was cut off
mid-structure. The control-character parse error is a consequence of the truncated tail,
not the root cause.

## Why no retry in this block
Rerunning 17C-R2 unchanged will truncate the same way. See
`MEDAI_17C_R2_NEXT_LIVE_STRATEGY_AFTER_R9.md`. `ready_for_another_live_retry=false`;
`requires_runner_strategy_change_before_retry=true`.

## Companion documents
- `MEDAI_17C_R2_PUBLIC_REPORT_REDACTION_POLICY_R9.md`
- `MEDAI_17C_R2_NEXT_LIVE_STRATEGY_AFTER_R9.md`
