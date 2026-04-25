# Phase 17 Dashboard

- run_id: `final_batch_50-20260425T140719051260_0000`
- timestamp: `2026-04-25T14:07:19.051260+00:00`
- dataset: `test_data\final_batch_50`
- attempted: `50`
- processed: `46`
- written: `45`
- written_with_review: `13`
- external_quota_blocked: `4`
- hard_failures: `0`
- avg_confidence: `0.692`
- duration_sec: `9.355`
- derived_queued_documents: `1`
- delta_written_vs_previous: `+0`
- delta_queued_vs_previous: `+0`
- route_distribution_requested: `{'gemini': 1, 'phi3': 1, 'spacy': 44, 'unknown': 4}`
- route_distribution_actual: `{'phi3': 1, 'spacy': 45}`
- review_counts: `{'clear': 13, 'quarantined': 13}`

## Inputs To Outputs

- Source artifacts: `C:\Users\S1\Documents\Codex\2026-04-22-connect-github\artifacts\phase12_real_world_validation\phase12_real_world_validation_summary.json` and `C:\Users\S1\Documents\Codex\2026-04-22-connect-github\reports\phase15\validation_aggregate.json`
- Input: latest validation summary + aggregate
- Decision layer: read-only metrics collector
- Output: appended run history + dashboard view
