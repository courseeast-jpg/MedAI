# Phase 21 Observability Report

- Generated at: `2026-04-26T03:13:49.855556+00:00`
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
- Run lock acquired: `True`
- Run lock released: `True`
- Stale lock recovered: `False`
- Retry eligible count: `4`
- Non-retryable failure count: `1`
- Timeout count: `0`
- Cleanup completed: `False`
- Production mode: `OFF`
- Production gate passed: `True`
- Production gate failed reason: `None`
- Dry run executed: `False`
- Controlled run limit applied: `False`
- Run blocked by gate: `False`

## Route Counts

- Extractor route counts: `{'phi3': 1, 'spacy': 45}`
- Extractor actual counts: `{'phi3': 1, 'spacy': 45}`

## Review Queue

- Review queue category counts: `{'external_quota_block': 4, 'truth_resolution_review': 26, 'validation_review': 1}`
- Review queue path: `C:\Users\S1\Documents\Codex\2026-04-22-connect-github\artifacts\phase12_real_world_validation\runtime\review_queue.jsonl`

## Stage Durations

- `consensus` -> records=46 events=46 total_ms=0.0 avg_ms=0.0 max_ms=0.0
- `extraction` -> records=50 events=96 total_ms=4474.544 avg_ms=89.491 max_ms=3060.053
- `final_write` -> records=39 events=124 total_ms=34785.118 avg_ms=891.926 max_ms=5150.657
- `language_support` -> records=50 events=50 total_ms=0.0 avg_ms=0.0 max_ms=0.0
- `medical_coding` -> records=46 events=46 total_ms=0.0 avg_ms=0.0 max_ms=0.0
- `safety_gate` -> records=26 events=98 total_ms=30962.591 avg_ms=1190.869 max_ms=4740.476
- `semantic_enrichment` -> records=46 events=46 total_ms=0.0 avg_ms=0.0 max_ms=0.0
- `truth_resolution` -> records=39 events=124 total_ms=34804.618 avg_ms=892.426 max_ms=5176.336
- `validation` -> records=46 events=46 total_ms=0.0 avg_ms=0.0 max_ms=0.0

## Stability Guardrails

- Determinism: `{'mode': 'deterministic_path', 'seed': None, 'ordering': 'sorted_pdf_listing'}`
- External quota blocks are counted separately from hard failures.
- Review queue counts are derived from the normalized JSONL queue written during validation.
