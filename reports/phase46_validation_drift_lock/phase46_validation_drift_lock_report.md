# Phase 46 Validation Drift Benchmark Lock

- Generated at: `2026-05-01T22:42:36.982167+00:00`
- Baseline manifest: `G:\Codex\2026-04-22-connect-github\validation_baselines\holdout_phase45_baseline.json`
- Live validation report: `G:\Codex\2026-04-22-connect-github\reports\phase45_cyrillic_nonlab_review\phase45_cyrillic_nonlab_review_report.json`
- Conclusion: `deterministic_lock_ready`

## Frozen Phase45 Baseline

- Baseline name: `holdout_phase45_baseline`
- Phase: `Phase 45 Cyrillic Non-Lab Review Classification Refinement`
- Commit: `e6e437d`
- Timestamp: `2026-05-01T22:15:52.450925+00:00`

## Count Comparison

| Metric | Expected | Live | Delta |
| --- | ---: | ---: | ---: |
| total_files | 8 | 8 | 0 |
| accepted | 2 | 2 | 0 |
| review | 6 | 6 | 0 |
| review_ocr_quality | 2 | 2 | 0 |
| empty | 2 | 2 | 0 |

## Runtime Drift

- runtime_drift_detected: `False`
- runtime_drift_files: `[]`
- unexpected_status_changes: `[]`
- accepted_due_to_current_phase: `[]`

## Safety Regression Section

- poor_ocr_became_accepted: `False`
- empty_extraction_became_accepted: `False`
- review_ocr_quality_became_accepted_without_approved_phase_reason: `False`
- confidence_gate_bypass_detected: `False`
- accepted_count_increased_unapproved: `False`
- phi_report_archive_or_review_paths_tracked: `False`
- safety_regression: `False`

## Per-file Comparison

| File | Expected | Live | Classification | Drift-sensitive | Safety | Reason |
| --- | --- | --- | --- | --- | --- | --- |
| Results 1.pdf | review | review | expected_match | yes | no | status_matches_frozen_baseline |
| Results 2.pdf | review | review | expected_match | no | no | status_matches_frozen_baseline |
| Test Results 2.pdf | review_ocr_quality | review_ocr_quality | expected_match | no | no | status_matches_frozen_baseline |
| Test Results 3.pdf | review | review | expected_match | no | no | status_matches_frozen_baseline |
| Test Results 4.pdf | accepted | accepted | expected_match | no | no | status_matches_frozen_baseline |
| Test Results 5.pdf | review_ocr_quality | review_ocr_quality | expected_match | no | no | status_matches_frozen_baseline |
| Test Results 6.pdf | review | review | expected_match | no | no | status_matches_frozen_baseline |
| Urinalysis, Routine.pdf | accepted | accepted | expected_match | no | no | status_matches_frozen_baseline |
