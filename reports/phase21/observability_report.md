# Phase 21 Observability Report

- Generated at: `2026-04-25T03:41:34.252300+00:00`
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
- Enrichment applied count: `46`
- Negation detected count: `0`
- Temporal detected count: `0`
- Relationships detected count: `0`

## Route Counts

- Extractor route counts: `{'spacy': 46}`
- Extractor actual counts: `{'spacy': 46}`

## Review Queue

- Review queue category counts: `{'external_quota_block': 4, 'truth_resolution_review': 26}`
- Review queue path: `C:\Users\S1\Documents\Codex\2026-04-22-connect-github\artifacts\phase12_real_world_validation\runtime\review_queue.jsonl`

## Stage Durations

- `consensus` -> records=46 events=46 total_ms=0.0 avg_ms=0.0 max_ms=0.0
- `extraction` -> records=50 events=96 total_ms=4174.064 avg_ms=83.481 max_ms=2960.413
- `final_write` -> records=39 events=128 total_ms=33712.548 avg_ms=864.424 max_ms=5094.306
- `safety_gate` -> records=26 events=102 total_ms=30114.733 avg_ms=1158.259 max_ms=4659.122
- `semantic_enrichment` -> records=46 events=46 total_ms=0.0 avg_ms=0.0 max_ms=0.0
- `truth_resolution` -> records=39 events=128 total_ms=33778.825 avg_ms=866.124 max_ms=5106.221
- `validation` -> records=46 events=46 total_ms=0.0 avg_ms=0.0 max_ms=0.0

## Stability Guardrails

- Determinism: `{'mode': 'deterministic_path', 'seed': None, 'ordering': 'sorted_pdf_listing'}`
- External quota blocks are counted separately from hard failures.
- Review queue counts are derived from the normalized JSONL queue written during validation.
