# Phase 12 Real-World Validation Summary

- Generated at: 2026-04-24T20:36:27.223157+00:00
- Dataset: `test_data\final_batch_50`
- Documents processed: 6/10
- Written: 6
- Queued for review: 0
- External quota blocked: 4
- Hard failures: 0
- Target window met (10-20 docs): True
- Run passed: True

## Aggregate

- Outcomes: {'written': 6}
- Validation statuses: {'accepted': 6}
- Extractors: {'spacy': 6}
- Average confidence: 0.7
- Total entities: 24
- Total written: 24
- Total queued: 0
- Total blocked: 0

## Runtime MKB

- Counts: {'total': 17, 'active': 5, 'hypothesis': 0, 'quarantined': 12}

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
