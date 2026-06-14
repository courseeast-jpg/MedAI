# 17C-R2 next live strategy (after R9)

- failure_class: `provider_truncated_by_max_tokens`
- provider_finish_category: `MAX_TOKENS`
- recommended_next_live_strategy: `split_output_or_reduce_schema`

Output exceeded the per-request token ceiling. Before a further live run, split the extraction schema into smaller sections (or raise maxOutputTokens only within the $0.40 cap) AND require strict JSON escaping; a rerun without changes repeats the same truncation.

## Hard gate before any next live run
- ready_for_another_live_retry: `false`
- requires_runner_strategy_change_before_retry: `true`
- Rerunning 17C-R2 unchanged will truncate the same way at the output-token ceiling.

## Concrete options (local design; not implemented in this block)
1. Split the extraction schema into smaller per-section requests (labs, diagnoses, medications, ...), each well under the output-token ceiling.
2. Or raise `maxOutputTokens` only within the $0.40 total / $0.05 per-chunk caps, paired with strict JSON escaping requirements and continued strict parser rejection.
3. Keep stop-on-first-failure, checkpoint resume, and failed-evidence preservation on.
