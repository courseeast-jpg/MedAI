# Phase 17 Dashboard

- run_id: `final_batch_50-20260424T220419424738_0000`
- timestamp: `2026-04-24T22:04:19.424738+00:00`
- dataset: `test_data\final_batch_50`
- attempted: `50`
- processed: `46`
- written: `46`
- written_with_review: `13`
- external_quota_blocked: `4`
- hard_failures: `0`
- avg_confidence: `0.700`
- duration_sec: `11.299`
- derived_queued_documents: `0`
- delta_written_vs_previous: `+0`
- delta_queued_vs_previous: `-1`
- route_distribution_requested: `{'gemini': 1, 'spacy': 45, 'unknown': 4}`
- route_distribution_actual: `{'spacy': 46}`
- review_counts: `{'clear': 13, 'quarantined': 13}`

## Inputs To Outputs

- Source artifacts: `C:\Users\S1\Documents\Codex\2026-04-22-connect-github\artifacts\phase12_real_world_validation\phase12_real_world_validation_summary.json` and `C:\Users\S1\Documents\Codex\2026-04-22-connect-github\reports\phase15\validation_aggregate.json`
- Input: latest validation summary + aggregate
- Decision layer: read-only metrics collector
- Output: appended run history + dashboard view
