# Phase 12 Real-World Validation Summary

- Generated at: 2026-04-25T13:18:33.660436+00:00
- Dataset: `test_data\final_batch_50`
- Documents processed: 50/50
- Written: 45
- Queued for review: 5
- External quota blocked: 0
- Hard failures: 0
- Review queue path: `C:\Users\S1\Documents\Codex\2026-04-22-connect-github\artifacts\phase12_real_world_validation\runtime\review_queue.jsonl`
- Review queue items: 29
- Target window met (10-20 docs): False
- Run passed: True

## Aggregate

- Outcomes: {'queued_for_review': 5, 'written': 33, 'written_with_review': 12}
- Validation statuses: {'accepted': 45, 'needs_review': 5}
- Extractors: {'phi3': 5, 'spacy': 45}
- Average confidence: 0.698
- Total entities: 122
- Total written: 100
- Total queued: 34
- Total blocked: 0
- Semantic enrichment applied: 50
- Negation detected count: 0
- Temporal detected count: 0
- Relationships detected count: 0

## Runtime MKB

- Counts: {'total': 48, 'active': 14, 'hypothesis': 0, 'quarantined': 34}

## Component State

- sql_store: True
- vector_store: False
- quality_gate: False
- medication_gate: False
- governance_active: True
- review_queue_path: C:\Users\S1\Documents\Codex\2026-04-22-connect-github\artifacts\phase12_real_world_validation\runtime\review_queue.jsonl
- audit_log_path: C:\Users\S1\Documents\Codex\2026-04-22-connect-github\artifacts\phase12_real_world_validation\runtime\execution_audit.jsonl
- stage_audit_log_path: C:\Users\S1\Documents\Codex\2026-04-22-connect-github\artifacts\phase12_real_world_validation\runtime\pipeline_stages.jsonl
- runtime_db_path: C:\Users\S1\Documents\Codex\2026-04-22-connect-github\artifacts\phase12_real_world_validation\runtime\mkb.db

## Recommendations

- Inspect review-queued documents and classify whether queues are expected governance behavior or extraction misses.

## Documents

- `long_noisy_01.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `long_noisy_02.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `long_noisy_03.pdf` -> status=processed outcome=queued_for_review validation=needs_review confidence=0.68
- `long_noisy_04.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `long_noisy_05.pdf` -> status=processed outcome=queued_for_review validation=needs_review confidence=0.68
- `long_noisy_06.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `long_noisy_07.pdf` -> status=processed outcome=queued_for_review validation=needs_review confidence=0.68
- `long_noisy_08.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `long_noisy_09.pdf` -> status=processed outcome=queued_for_review validation=needs_review confidence=0.68
- `long_noisy_10.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `short_01.pdf` -> status=processed outcome=queued_for_review validation=needs_review confidence=0.68
- `short_02.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `short_03.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `short_04.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `short_05.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `short_06.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `short_07.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `short_08.pdf` -> status=processed outcome=written_with_review validation=accepted confidence=0.7
- `short_09.pdf` -> status=processed outcome=written_with_review validation=accepted confidence=0.7
- `short_10.pdf` -> status=processed outcome=written_with_review validation=accepted confidence=0.7
- `short_11.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `short_12.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `short_13.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `short_14.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `short_15.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `short_16.pdf` -> status=processed outcome=written_with_review validation=accepted confidence=0.7
- `short_17.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `short_18.pdf` -> status=processed outcome=written_with_review validation=accepted confidence=0.7
- `short_19.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `short_20.pdf` -> status=processed outcome=written_with_review validation=accepted confidence=0.7
- `short_21.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `short_22.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `short_23.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `short_24.pdf` -> status=processed outcome=written_with_review validation=accepted confidence=0.7
- `short_25.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `short_26.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `short_27.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `short_28.pdf` -> status=processed outcome=written_with_review validation=accepted confidence=0.7
- `short_29.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `short_30.pdf` -> status=processed outcome=written_with_review validation=accepted confidence=0.7
- `short_31.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `short_32.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `short_33.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `short_34.pdf` -> status=processed outcome=written_with_review validation=accepted confidence=0.7
- `short_35.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `short_36.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `short_37.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `short_38.pdf` -> status=processed outcome=written_with_review validation=accepted confidence=0.7
- `short_39.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `short_40.pdf` -> status=processed outcome=written_with_review validation=accepted confidence=0.7
