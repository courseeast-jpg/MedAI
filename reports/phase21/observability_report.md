# Phase 21 Observability Report

- Generated at: `2026-04-25T02:31:12.027145+00:00`
- Dataset: `test_data\final_batch_50`
- Attempted documents: `50`
- Processed documents: `46`
- Written documents: `45`
- Queued for review documents: `1`
- Review queue items: `31`
- External quota blocked: `4`
- Hard failures: `0`
- Average confidence: `0.7`
- Route mismatch count: `2`
- Low-confidence count: `1`
- Quota-safe block count: `4`

## Route Counts

- Extractor route counts: `{'phi3': 1, 'spacy': 45}`
- Extractor actual counts: `{'phi3': 1, 'spacy': 45}`

## Review Queue

- Review queue category counts: `{'external_quota_block': 4, 'truth_resolution_review': 26, 'validation_review': 1}`
- Review queue path: `C:\Users\S1\Documents\Codex\2026-04-22-connect-github\artifacts\phase12_real_world_validation\runtime\review_queue.jsonl`

## Stage Durations

- `consensus` -> records=46 events=46 total_ms=0.0 avg_ms=0.0 max_ms=0.0
- `extraction` -> records=50 events=96 total_ms=8825.074 avg_ms=176.501 max_ms=5065.951
- `final_write` -> records=41 events=126 total_ms=51840.436 avg_ms=1264.401 max_ms=9713.009
- `safety_gate` -> records=26 events=98 total_ms=48301.387 avg_ms=1857.746 max_ms=9282.912
- `truth_resolution` -> records=39 events=124 total_ms=51872.563 avg_ms=1330.066 max_ms=9725.01
- `validation` -> records=46 events=46 total_ms=0.0 avg_ms=0.0 max_ms=0.0

## Stability Guardrails

- Determinism: `{'mode': 'deterministic_path', 'seed': None, 'ordering': 'sorted_pdf_listing'}`
- External quota blocks are counted separately from hard failures.
- Review queue counts are derived from the normalized JSONL queue written during validation.
