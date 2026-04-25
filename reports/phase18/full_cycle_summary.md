# Phase 18 Full Cycle Summary

- Generated at: `2026-04-25T02:31:14.809768+00:00`
- Commit hash: `23818a0b9345ff5707b0ce83710b7eb333a39de3`
- Git status: `dirty`
- Success: `True`
- Failed step: `None`
- Test result: `171 passed`
- Phase 11 audit result: `passed`
- Validation attempted: `50`
- Validation processed: `46`
- Validation written: `45`
- Validation queued_for_review: `1`
- Validation external_quota_blocked: `4`
- Validation hard_failures: `0`
- Validation avg_confidence: `0.7`
- Validation review_queue_items: `31`
- Validation review_queue_path: `C:\Users\S1\Documents\Codex\2026-04-22-connect-github\artifacts\phase12_real_world_validation\runtime\review_queue.jsonl`
- Observability route_mismatch_count: `2`
- Observability low_confidence_count: `1`
- Observability quota_safe_block_count: `4`
- Observability metrics_path: `C:\Users\S1\Documents\Codex\2026-04-22-connect-github\artifacts\phase21\observability_metrics.json`
- Observability report_path: `C:\Users\S1\Documents\Codex\2026-04-22-connect-github\reports\phase21\observability_report.md`
- Calibration average_raw_confidence: `0.7`
- Calibration average_calibrated_confidence: `0.7`
- Calibration confidence_band_counts: `{'acceptable': 45, 'review': 1}`
- Calibration review_recommendation_counts: `{'accept': 44, 'accept_with_route_audit': 1, 'operator_review': 1}`
- Calibration route_mismatch_count: `2`
- Calibration metrics_path: `C:\Users\S1\Documents\Codex\2026-04-22-connect-github\artifacts\phase22\confidence_calibration.json`
- Calibration report_path: `C:\Users\S1\Documents\Codex\2026-04-22-connect-github\reports\phase22\accuracy_calibration_report.md`
- Dashboard export path: `C:\Users\S1\Documents\Codex\2026-04-22-connect-github\reports\phase17\dashboard_latest.md`
- Stability report path: `C:\Users\S1\Documents\Codex\2026-04-22-connect-github\reports\phase19\stability_report.md`
- Duration seconds: `262.337`

## Steps

- `tests` -> returncode=0 command=C:\Program Files\Python311\python.exe -m pytest tests
- `phase11_audit` -> returncode=0 command=C:\Program Files\Python311\python.exe scripts\run_phase11_integration_audit.py
- `validation` -> returncode=0 command=C:\Program Files\Python311\python.exe scripts\run_phase12_real_world_validation.py --dataset-dir test_data\final_batch_50 --quota-safe
- `dashboard_latest` -> returncode=0 command=C:\Program Files\Python311\python.exe scripts\run_phase17_dashboard.py --latest
- `dashboard_export` -> returncode=0 command=C:\Program Files\Python311\python.exe scripts\run_phase17_dashboard.py --export
