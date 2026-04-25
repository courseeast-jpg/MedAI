# Phase 19 Stability Report

- Overall status: `STABLE`
- Tolerances: `{'processed_delta_max': 2.0, 'written_delta_max': 2.0, 'queued_delta_max': 2.0, 'confidence_delta_max': 0.05}`

## Recent Runs

- `final_batch_50-20260425T130248248795_0000` -> processed=50 written=40 queued=10 quota=0 avg_conf=0.676
- `final_batch_50-20260425T131139698789_0000` -> processed=49 written=46 queued=3 quota=1 avg_conf=0.699
- `final_batch_50-20260425T131833660436_0000` -> processed=50 written=45 queued=5 quota=0 avg_conf=0.698

## Comparisons

- `final_batch_50-20260425T130248248795_0000` -> `final_batch_50-20260425T131139698789_0000` status=UNSTABLE deltas={'processed': -1, 'written': 6, 'queued_for_review': -7, 'quota_blocked': 1, 'avg_confidence': 0.023}
  explanation: Quota-blocked documents changed by 1, which can shift processed totals without any pipeline behavior change.
  explanation: Derived queued document count changed by -7; inspect review counts and quota behavior for this run.
  explanation: Average confidence changed by +0.023; compare document mix and quota-skipped documents.
- `final_batch_50-20260425T131139698789_0000` -> `final_batch_50-20260425T131833660436_0000` status=STABLE deltas={'processed': 1, 'written': -1, 'queued_for_review': 2, 'quota_blocked': -1, 'avg_confidence': -0.001}
  explanation: Quota-blocked documents changed by -1, which can shift processed totals without any pipeline behavior change.
  explanation: Derived queued document count changed by 2; inspect review counts and quota behavior for this run.
  explanation: Average confidence changed by -0.001; compare document mix and quota-skipped documents.
