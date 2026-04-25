# Phase 19 Stability Report

- Overall status: `STABLE`
- Tolerances: `{'processed_delta_max': 2.0, 'written_delta_max': 2.0, 'queued_delta_max': 2.0, 'confidence_delta_max': 0.05}`

## Recent Runs

- `final_batch_50-20260425T135722643897_0000` -> processed=46 written=46 queued=0 quota=4 avg_conf=0.700
- `final_batch_50-20260425T140220152047_0000` -> processed=46 written=45 queued=1 quota=4 avg_conf=0.692
- `final_batch_50-20260425T140719051260_0000` -> processed=46 written=45 queued=1 quota=4 avg_conf=0.692

## Comparisons

- `final_batch_50-20260425T135722643897_0000` -> `final_batch_50-20260425T140220152047_0000` status=STABLE deltas={'processed': 0, 'written': -1, 'queued_for_review': 1, 'quota_blocked': 0, 'avg_confidence': -0.008}
  explanation: Derived queued document count changed by 1; inspect review counts and quota behavior for this run.
  explanation: Average confidence changed by -0.008; compare document mix and quota-skipped documents.
- `final_batch_50-20260425T140220152047_0000` -> `final_batch_50-20260425T140719051260_0000` status=STABLE deltas={'processed': 0, 'written': 0, 'queued_for_review': 0, 'quota_blocked': 0, 'avg_confidence': 0.0}
  explanation: No material variance detected between the compared runs.
