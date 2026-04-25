# Phase 21 Observability Report

- Generated at: `2026-04-25T13:18:33.660436+00:00`
- Dataset: `test_data\final_batch_50`
- Attempted documents: `50`
- Processed documents: `50`
- Written documents: `45`
- Queued for review documents: `5`
- Review queue items: `29`
- External quota blocked: `0`
- Hard failures: `0`
- Average confidence: `0.698`
- Route mismatch count: `10`
- Low-confidence count: `5`
- Quota-safe block count: `0`
- Enrichment applied count: `50`
- Negation detected count: `0`
- Temporal detected count: `0`
- Relationships detected count: `0`

## Route Counts

- Extractor route counts: `{'phi3': 5, 'spacy': 45}`
- Extractor actual counts: `{'phi3': 5, 'spacy': 45}`

## Review Queue

- Review queue category counts: `{'truth_resolution_review': 24, 'validation_review': 5}`
- Review queue path: `C:\Users\S1\Documents\Codex\2026-04-22-connect-github\artifacts\phase12_real_world_validation\runtime\review_queue.jsonl`

## Stage Durations

- `consensus` -> records=50 events=50 total_ms=0.0 avg_ms=0.0 max_ms=0.0
- `extraction` -> records=50 events=100 total_ms=44165.982 avg_ms=883.32 max_ms=7415.505
- `final_write` -> records=48 events=134 total_ms=182827.805 avg_ms=3808.913 max_ms=44355.948
- `safety_gate` -> records=26 events=100 total_ms=177468.566 avg_ms=6825.714 max_ms=41427.605
- `semantic_enrichment` -> records=50 events=50 total_ms=0.0 avg_ms=0.0 max_ms=0.0
- `truth_resolution` -> records=38 events=124 total_ms=183066.499 avg_ms=4817.539 max_ms=44367.731
- `validation` -> records=50 events=50 total_ms=0.0 avg_ms=0.0 max_ms=0.0

## Stability Guardrails

- Determinism: `{'mode': 'deterministic_path', 'seed': None, 'ordering': 'sorted_pdf_listing'}`
- External quota blocks are counted separately from hard failures.
- Review queue counts are derived from the normalized JSONL queue written during validation.
