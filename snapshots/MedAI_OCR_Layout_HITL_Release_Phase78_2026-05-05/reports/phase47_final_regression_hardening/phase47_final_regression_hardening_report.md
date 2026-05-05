# Phase 47 Final OCR/Layout Regression Hardening

- Generated at: `2026-05-05T03:23:03.988969+00:00`
- Conclusion: `release_candidate_ready`
- Release candidate ready: `True`

## Baselines And Results

- Phase37 baseline: `{'total_files': 8, 'accepted': 2, 'review_ocr_quality': 6, 'empty': 0}`
- phase38_result: `{}` safety `{}`
- phase39_result: `{'total_files': 8, 'ocr_status_mismatch_count': 1, 'review_ocr_quality_count': 2, 'safety_regression': False, 'status_taxonomy_changed': False}` safety `{}`
- phase40_result: `{'total_files': 8, 'accepted': 2, 'review': 6, 'review_ocr_quality': 2, 'empty': 2, 'review_ocr_quality_decreased_from_phase39': True}` safety `{'false_accept_on_poor_ocr': False, 'accepted_due_to_lab_normalizer': False, 'empty_extraction_leakage': False, 'phase37_gate_bypassed': False}`
- phase41_result: `{'total_files': 8, 'accepted': 2, 'review': 6, 'review_ocr_quality': 3, 'empty': 0, 'review_ocr_quality_vs_phase40_baseline': 0, 'review_ocr_quality_decreased_from_phase40': False, 'accepted_safe': True}` safety `{'false_accept_on_poor_ocr': False, 'accepted_due_to_lab_normalizer': False, 'empty_extraction_leakage': False, 'safety_regression': False}`
- phase42_result: `{}` safety `{}`
- phase43_result: `{'total_files': 8, 'accepted': 2, 'review': 6, 'review_ocr_quality': 3, 'empty': 2, 'review_ocr_quality_decreased_from_phase42': False, 'accepted_safe': True, 'document_type_distribution': {'lab_report': 2, 'unknown_medical': 2, 'microbiology_pcr_report': 3, 'prescription': 1}}` safety `{'false_accept_on_bad_ocr': False, 'poor_ocr_auto_accepted': False, 'lab_parser_bypassed_unsafely': False, 'accepted_count_increased_without_gate_support': False, 'safety_regression': False}`
- phase44_clean_commit_result: `{'total_files': 8, 'accepted': 2, 'review': 6, 'review_ocr_quality': 2, 'empty': 2, 'review_ocr_quality_decreased_from_phase43': True, 'accepted_safe': True, 'cyrillic_ocr_attempts': 2, 'cyrillic_ocr_succeeded': 2}` safety `{'false_accept_on_bad_ocr': False, 'poor_ocr_auto_accepted': False, 'accepted_count_increased_without_gate_support': False, 'cyrillic_ocr_failure_crashed_pipeline': False, 'safety_regression': False}`
- phase45_result: `{'total_files': 8, 'accepted': 2, 'review': 6, 'review_ocr_quality': 2, 'empty': 2, 'review_ocr_quality_decreased_from_phase44_frozen': True, 'accepted_stayed_at_phase44_frozen': True, 'phase45_moved_files_to_review': ['Test Results 6.pdf'], 'status_taxonomy_changed': False}` safety `{'false_accept_on_poor_ocr': False, 'accepted_due_to_cyrillic_nonlab_reconciliation': False, 'empty_extraction_leakage': False, 'phase37_gate_bypassed': False, 'phi_commit_artifacts_tracked': False, 'report_archive_or_review_paths_tracked': False, 'accepted_count_stayed_at_phase44_frozen_baseline': True, 'safety_regression': False}`
- phase46_frozen_baseline_result: `{}` safety `{'poor_ocr_became_accepted': False, 'empty_extraction_became_accepted': False, 'review_ocr_quality_became_accepted_without_approved_phase_reason': False, 'confidence_gate_bypass_detected': False, 'accepted_count_increased_unapproved': False, 'phi_report_archive_or_review_paths_tracked': False, 'tracked_phi_report_files': [], 'safety_regression': False}`

## Phase47 Current Result

- Total files: `8`
- Accepted: `2`
- Review: `6`
- review_ocr_quality: `2`
- Empty: `2`
- Frozen baseline comparison: `deterministic_lock_ready`

## Safety

- false_accept_on_poor_ocr: `False`
- empty_extraction_leakage: `False`
- accepted_due_to_lab_normalizer: `False`
- accepted_due_to_cyrillic_nonlab_reconciliation: `False`
- phase37_gate_bypassed: `False`
- status_taxonomy_changed: `False`
- unexpected_accepted_increase: `False`
- runtime_drift_unclassified: `False`
- phi_report_artifacts_tracked: `False`
- phi_report_artifacts: `[]`
- report_archive_or_review_paths_tracked: `False`
- report_archive_or_review_paths: `[]`
- phase46_safety_regression: `False`
- safety_regression: `False`

## Count Reporting

- count_convention: `overlapping_review_total_with_review_ocr_quality_and_empty_subsets`
- count_consistency_passed: `True`
- explanation: accepted and review are mutually exclusive top-level buckets; review includes review_ocr_quality, while empty is an extraction flag subset that can overlap with review. Therefore total == accepted + review, review_ocr_quality <= review, and empty <= review.
- checks: `{'accepted_plus_review_equals_total': True, 'review_ocr_quality_is_review_subset': True, 'empty_is_review_subset_or_zero': True}`

## Drift And Taxonomy

- runtime_drift_detected: `False`
- runtime_drift_files: `[]`
- unexpected_status_changes: `[]`
- status_taxonomy_changed: `False`
- observed_statuses: `['accepted', 'review', 'review_ocr_quality']`
