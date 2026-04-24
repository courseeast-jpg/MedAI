# Phase 18 One-Command Operation

## Windows Launcher

Run:

```bat
run_medai_phase18.bat
```

This executes, in order:

1. `python -m pytest tests`
2. `python scripts\run_phase11_integration_audit.py`
3. `python scripts\run_phase12_real_world_validation.py --dataset-dir test_data\final_batch_50 --quota-safe`
4. `python scripts\run_phase17_dashboard.py --latest`
5. `python scripts\run_phase17_dashboard.py --export`

If any step fails, the launcher stops immediately, prints the failed step, and exits nonzero.

## Python Wrapper

Run:

```bat
python scripts\run_phase18_full_cycle.py
```

This runs the same sequence and writes:

- `reports/phase18/full_cycle_summary.json`
- `reports/phase18/full_cycle_summary.md`

## Expected Outputs

- Updated validation artifacts under `artifacts/phase12_real_world_validation`
- Updated Phase 13 and Phase 15 reports
- Updated Phase 17 dashboard artifacts:
  - `reports/phase17/run_history.jsonl`
  - `reports/phase17/dashboard_latest.md`
- New Phase 18 wrapper summary:
  - `reports/phase18/full_cycle_summary.json`
  - `reports/phase18/full_cycle_summary.md`

## If A Step Fails

- Read the failing step name from the launcher output or `failed_step` in the Phase 18 summary.
- Fix the failed step directly; do not skip later steps.
- Re-run the same command that failed first.
- After it succeeds, rerun the full wrapper so the reports and dashboard stay consistent.
