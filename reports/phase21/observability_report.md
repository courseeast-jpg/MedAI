# Phase 21 Observability Report

- Generated at: `2026-04-25T03:11:54.109934+00:00`
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
- `extraction` -> records=50 events=96 total_ms=3927.788 avg_ms=78.556 max_ms=2728.869
- `final_write` -> records=39 events=128 total_ms=32629.339 avg_ms=836.65 max_ms=4855.346
- `safety_gate` -> records=26 events=102 total_ms=29052.954 avg_ms=1117.421 max_ms=4428.241
- `truth_resolution` -> records=39 events=128 total_ms=32663.191 avg_ms=837.518 max_ms=4863.337
- `validation` -> records=46 events=46 total_ms=0.0 avg_ms=0.0 max_ms=0.0

## Stability Guardrails

- Determinism: `{'mode': 'deterministic_path', 'seed': None, 'ordering': 'sorted_pdf_listing'}`
- External quota blocks are counted separately from hard failures.
- Review queue counts are derived from the normalized JSONL queue written during validation.
