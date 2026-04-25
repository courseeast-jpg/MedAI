# Phase 19 Stability Report

- Overall status: `STABLE`
- Tolerances: `{'processed_delta_max': 2.0, 'written_delta_max': 2.0, 'queued_delta_max': 2.0, 'confidence_delta_max': 0.05}`

## Recent Runs

- `final_batch_50-20260425T004734195677_0000` -> processed=46 written=46 queued=0 quota=4 avg_conf=0.700
- `final_batch_50-20260425T005831860752_0000` -> processed=47 written=46 queued=1 quota=3 avg_conf=0.700
- `final_batch_50-20260425T012114151373_0000` -> processed=46 written=46 queued=0 quota=4 avg_conf=0.700

## Comparisons

- `final_batch_50-20260425T004734195677_0000` -> `final_batch_50-20260425T005831860752_0000` status=STABLE deltas={'processed': 1, 'written': 0, 'queued_for_review': 1, 'quota_blocked': -1, 'avg_confidence': 0.0}
  explanation: Quota-blocked documents changed by -1, which can shift processed totals without any pipeline behavior change.
  explanation: Derived queued document count changed by 1; inspect review counts and quota behavior for this run.
- `final_batch_50-20260425T005831860752_0000` -> `final_batch_50-20260425T012114151373_0000` status=STABLE deltas={'processed': -1, 'written': 0, 'queued_for_review': -1, 'quota_blocked': 1, 'avg_confidence': 0.0}
  explanation: Quota-blocked documents changed by 1, which can shift processed totals without any pipeline behavior change.
  explanation: Derived queued document count changed by -1; inspect review counts and quota behavior for this run.
