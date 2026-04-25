# Phase 21 Observability Report

- Generated at: `2026-04-25T01:21:14.151373+00:00`
- Dataset: `test_data\final_batch_50`
- Attempted documents: `50`
- Processed documents: `46`
- Written documents: `46`
- Queued for review documents: `0`
- Review queue items: `30`
- External quota blocked: `4`
- Hard failures: `0`
- Average confidence: `0.7`
- Route mismatch count: `1`
- Low-confidence count: `0`
- Quota-safe block count: `4`

## Route Counts

- Extractor route counts: `{'spacy': 46}`
- Extractor actual counts: `{'spacy': 46}`

## Review Queue

- Review queue category counts: `{'external_quota_block': 4, 'truth_resolution_review': 26}`
- Review queue path: `C:\Users\S1\Documents\Codex\2026-04-22-connect-github\artifacts\phase12_real_world_validation\runtime\review_queue.jsonl`

## Stage Durations

- `consensus` -> records=46 events=46 total_ms=0.0 avg_ms=0.0 max_ms=0.0
- `extraction` -> records=50 events=96 total_ms=3926.52 avg_ms=78.53 max_ms=2663.389
- `final_write` -> records=39 events=128 total_ms=34900.624 avg_ms=894.888 max_ms=4782.868
- `safety_gate` -> records=26 events=102 total_ms=30724.759 avg_ms=1181.721 max_ms=4368.87
- `truth_resolution` -> records=39 events=128 total_ms=34932.855 avg_ms=895.714 max_ms=4796.965
- `validation` -> records=46 events=46 total_ms=0.0 avg_ms=0.0 max_ms=0.0

## Stability Guardrails

- Determinism: `{'mode': 'deterministic_path', 'seed': None, 'ordering': 'sorted_pdf_listing'}`
- External quota blocks are counted separately from hard failures.
- Review queue counts are derived from the normalized JSONL queue written during validation.
