# 17C-R2 Next Live Strategy (after R9)

## Failure class
`provider_truncated_by_max_tokens` — the model's JSON output hit the per-request
output-token ceiling (`finishReason=MAX_TOKENS`, `candidatesTokenCount=2048`) and was cut
off mid-object.

## Hard gate before any next live run
- `ready_for_another_live_retry = false`
- `requires_runner_strategy_change_before_retry = true`
- Rerunning 17C-R2 unchanged will truncate the same way at the output-token ceiling.

## Recommended strategy: `split_output_or_reduce_schema`
Pick one (local design; **not** implemented in this block):

1. **Split the schema into smaller per-section requests** — e.g. labs, diagnoses,
   medications, procedures, imaging, pathology each as its own request, every response
   well under the output ceiling. Reassemble locally. Lowest truncation risk.
2. **Raise `maxOutputTokens`** only within the authorized caps ($0.05/chunk, $0.40 total),
   paired with strict JSON escaping requirements in the prompt and continued strict parser
   rejection. Simpler but still risks truncation on the largest documents.

In both cases keep: stop-on-first-failure, checkpoint resume (R8), and failed-evidence
preservation (R8). Strengthen the prompt's escaping requirement so control characters in
the source do not surface unescaped in the JSON body.

## Out of scope here
No live retry is implemented in this block. The next block should implement the chosen
runner strategy change, then re-validate the entry gate before any authorized live run.
