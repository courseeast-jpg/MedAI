# Phase 17 Dashboard

- run_id: `final_batch_50-20260425T131833660436_0000`
- timestamp: `2026-04-25T13:18:33.660436+00:00`
- dataset: `test_data\final_batch_50`
- attempted: `50`
- processed: `50`
- written: `45`
- written_with_review: `12`
- external_quota_blocked: `0`
- hard_failures: `0`
- avg_confidence: `0.698`
- duration_sec: `48.676`
- derived_queued_documents: `5`
- delta_written_vs_previous: `-1`
- delta_queued_vs_previous: `+2`
- route_distribution_requested: `{'gemini': 10, 'spacy': 40}`
- route_distribution_actual: `{'phi3': 5, 'spacy': 45}`
- review_counts: `{'clear': 12, 'quarantined': 12}`

## Inputs To Outputs

- Source artifacts: `C:\Users\S1\Documents\Codex\2026-04-22-connect-github\artifacts\phase12_real_world_validation\phase12_real_world_validation_summary.json` and `C:\Users\S1\Documents\Codex\2026-04-22-connect-github\reports\phase15\validation_aggregate.json`
- Input: latest validation summary + aggregate
- Decision layer: read-only metrics collector
- Output: appended run history + dashboard view
