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

## Notes

- This runner is validation-only.
- It uses an isolated runtime SQLite database under the Phase 12 artifact directory.
- It does not rebuild or refactor the pipeline.
