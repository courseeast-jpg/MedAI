# Phase 27 Production Hardening Report

- Generated at: `2026-04-26T03:13:49.855556+00:00`
- Dataset: `test_data\final_batch_50`
- Attempted documents: `50`
- Processed documents: `46`
- Written documents: `45`
- Queued for review documents: `1`
- External quota blocked: `4`
- Hard failures: `0`
- Run ID: `phase12_validation-34dc9ad511d46ada`
- Script name: `run_phase12_real_world_validation.py`
- Lock path: `C:\Users\S1\Documents\Codex\2026-04-22-connect-github\artifacts\phase27\validation_run.lock`
- Run lock acquired: `True`
- Run lock released: `True`
- Stale lock recovered: `False`
- Retry eligible count: `4`
- Non-retryable failure count: `1`
- Timeout count: `0`
- Cleanup completed: `False`
- Failure category counts: `{'external_quota_block': 4, 'none': 45, 'operator_review_required': 1}`

## Runtime Guardrails

- Determinism: `{'mode': 'deterministic_path', 'seed': None, 'ordering': 'sorted_pdf_listing'}`
- Runtime hardening is script-level only and does not alter extraction, routing, confidence, review, enrichment, coding, or language outputs.
- External quota blocks and operator-review outcomes remain non-hard-failure categories.
- The single-run lock rejects concurrent overlap and allows deterministic stale-lock recovery with safe cleanup.
