# Phase 12 Real-World Validation

Phase 12 starts from the locked Phase 10-11 baseline and validates real document behavior without rebuilding the pipeline.

## Scope

- use the existing execution pipeline
- process a 10-20 document PDF batch
- record actual outcomes: written, queued for review, blocked, or failed
- preserve governance-active behavior

## Runner

```powershell
python scripts\run_phase12_real_world_validation.py --dataset-dir test_data\final_batch_50 --limit 10
```

## Artifacts

- `artifacts/phase12_real_world_validation/phase12_real_world_validation_summary.json`
- `artifacts/phase12_real_world_validation/phase12_real_world_validation_summary.md`
- `artifacts/phase12_real_world_validation/phase12_real_world_validation_documents.jsonl`
- `artifacts/phase12_real_world_validation/runtime/review_queue.jsonl`

## Review Queue

An item is written to the review queue when the pipeline cannot deterministically finalize it:

- extraction validation returns `needs_review` or `rejected`
- truth resolution quarantines a fact
- medication safety requires operator review
- `--quota-safe` classifies an external quota error as `external_quota_blocked`

Each queue record captures:

- `run_id`
- `document_id` or source filename
- `reason`
- `confidence`
- `extractor_route`
- `extractor_actual`
- `timestamp`
- `recommended_action`
- `raw_evidence_path` when available

This supports auditability by separating operator follow-up from hard-failure accounting. Quota-safe blocks stay visible in the queue and summary, but they do not become hard failures.

## Notes

- This runner is validation-only.
- It uses an isolated runtime SQLite database under the Phase 12 artifact directory.
- It does not rebuild or refactor the pipeline.
