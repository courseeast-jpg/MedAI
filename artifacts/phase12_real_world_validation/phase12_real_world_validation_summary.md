# Phase 12 Real-World Validation Summary

- Generated at: 2026-04-24T22:04:19.424738+00:00
- Dataset: `test_data\final_batch_50`
- Documents processed: 46/50
- Written: 46
- Queued for review: 0
- External quota blocked: 4
- Hard failures: 0
- Target window met (10-20 docs): False
- Run passed: True

## Aggregate

- Outcomes: {'written': 33, 'written_with_review': 13}
- Validation statuses: {'accepted': 46}
- Extractors: {'spacy': 46}
- Average confidence: 0.7
- Total entities: 115
- Total written: 102
- Total queued: 26
- Total blocked: 0

## Runtime MKB

- Counts: {'total': 184, 'active': 14, 'hypothesis': 0, 'quarantined': 170}

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

- Some documents were skipped due to external API quota exhaustion; rerun later to complete the sample without changing pipeline behavior.
- Inspect review-queued documents and classify whether queues are expected governance behavior or extraction misses.

## Documents

- `long_noisy_01.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `long_noisy_02.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `long_noisy_03.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `long_noisy_04.pdf` -> status=external_quota_blocked outcome=external_quota_blocked validation=skipped_external_quota confidence=0.0
- `long_noisy_05.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `long_noisy_06.pdf` -> status=external_quota_blocked outcome=external_quota_blocked validation=skipped_external_quota confidence=0.0
- `long_noisy_07.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `long_noisy_08.pdf` -> status=external_quota_blocked outcome=external_quota_blocked validation=skipped_external_quota confidence=0.0
- `long_noisy_09.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `long_noisy_10.pdf` -> status=external_quota_blocked outcome=external_quota_blocked validation=skipped_external_quota confidence=0.0
- `short_01.pdf` -> status=processed outcome=written_with_review validation=accepted confidence=0.7
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
- `short_16.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
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
- `short_36.pdf` -> status=processed outcome=written_with_review validation=accepted confidence=0.7
- `short_37.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `short_38.pdf` -> status=processed outcome=written_with_review validation=accepted confidence=0.7
- `short_39.pdf` -> status=processed outcome=written validation=accepted confidence=0.7
- `short_40.pdf` -> status=processed outcome=written_with_review validation=accepted confidence=0.7
