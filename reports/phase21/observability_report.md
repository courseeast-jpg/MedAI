# Phase 21 Observability Report

- Generated at: `2026-04-25T15:20:49.049544+00:00`
- Dataset: `test_data\final_batch_50`
- Attempted documents: `50`
- Processed documents: `46`
- Written documents: `45`
- Queued for review documents: `1`
- Review queue items: `31`
- External quota blocked: `4`
- Hard failures: `0`
- Average confidence: `0.692`
- Route mismatch count: `1`
- Low-confidence count: `1`
- Quota-safe block count: `4`
- Enrichment applied count: `45`
- Negation detected count: `0`
- Temporal detected count: `0`
- Relationships detected count: `0`
- Coding attempted count: `82`
- Coding success count: `8`
- Coding unmapped count: `74`
- Coding ambiguous count: `0`
- Coding skipped count: `29`
- Language detected counts: `{'english': 46}`
- Cyrillic detected count: `0`
- Mixed language count: `0`
- Pending translation count: `0`
- Requires OCR count: `0`
- Language unknown count: `0`

## Route Counts

- Extractor route counts: `{'phi3': 1, 'spacy': 45}`
- Extractor actual counts: `{'phi3': 1, 'spacy': 45}`

## Review Queue

- Review queue category counts: `{'external_quota_block': 4, 'truth_resolution_review': 26, 'validation_review': 1}`
- Review queue path: `C:\Users\S1\Documents\Codex\2026-04-22-connect-github\artifacts\phase12_real_world_validation\runtime\review_queue.jsonl`

## Stage Durations

- `consensus` -> records=46 events=46 total_ms=0.0 avg_ms=0.0 max_ms=0.0
- `extraction` -> records=50 events=96 total_ms=3622.413 avg_ms=72.448 max_ms=2382.12
- `final_write` -> records=39 events=124 total_ms=31480.512 avg_ms=807.193 max_ms=4550.862
- `language_support` -> records=50 events=50 total_ms=0.0 avg_ms=0.0 max_ms=0.0
- `medical_coding` -> records=46 events=46 total_ms=0.0 avg_ms=0.0 max_ms=0.0
- `safety_gate` -> records=26 events=98 total_ms=27884.254 avg_ms=1072.471 max_ms=4126.598
- `semantic_enrichment` -> records=46 events=46 total_ms=0.0 avg_ms=0.0 max_ms=0.0
- `truth_resolution` -> records=39 events=124 total_ms=31509.536 avg_ms=807.937 max_ms=4560.696
- `validation` -> records=46 events=46 total_ms=0.0 avg_ms=0.0 max_ms=0.0

## Stability Guardrails

- Determinism: `{'mode': 'deterministic_path', 'seed': None, 'ordering': 'sorted_pdf_listing'}`
- External quota blocks are counted separately from hard failures.
- Review queue counts are derived from the normalized JSONL queue written during validation.
