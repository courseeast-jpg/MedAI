# Phase 45 Cyrillic Non-Lab Review Classification Refinement

- Generated at: `2026-05-01T22:56:18.268190+00:00`

## Frozen Baselines

- Phase37: `{'total_files': 8, 'accepted': 2, 'review_ocr_quality': 6, 'empty': 0}`
- Phase38: `{'total_files': 8, 'accepted': 2, 'review': 6, 'review_ocr_quality': 4, 'empty': 0}`
- Phase39: `{'total_files': 8, 'ocr_status_mismatches': 3, 'review_ocr_quality': 4, 'safety_regression': False}`
- Phase40: `{'total_files': 8, 'accepted': 2, 'review_ocr_quality': 3, 'empty': 0, 'safety_regression': False}`
- Phase43: `{'total_files': 8, 'accepted': 2, 'review_ocr_quality': 3, 'safety_regression': False}`
- Phase44 (FROZEN): `{'total_files': 8, 'accepted': 2, 'review_ocr_quality': 3, 'tests_passed': 378, 'safety_regression': False, 'clean_commit': '78d5c8d44be5cac568f4c05870acaa86ee7d573a'}`

## Phase 45 Result

- Total files: `8`
- Accepted: `2`
- Review: `6`
- review_ocr_quality: `2`
- Empty: `2`
- review_ocr_quality decreased from Phase44 frozen baseline (3): `True`
- accepted stayed at Phase44 frozen baseline (2): `True`
- Files moved review_ocr_quality -> review by Phase 45: `['Test Results 6.pdf']`
- Status taxonomy changed: `False`

## Runtime Drift

- runtime_drift_detected: `False`
- runtime_drift_files: `[]`
- runtime_drift_interpretation: "upstream extractor nondeterminism unrelated to Cyrillic Phase45 scope"

## Safety

- false_accept_on_poor_ocr: `False`
- accepted_due_to_cyrillic_nonlab_reconciliation: `False`
- empty_extraction_leakage: `False`
- phase37_gate_bypassed: `False`
- phi_commit_artifacts_tracked: `False`
- report_archive_or_review_paths_tracked: `False`
- accepted_count_stayed_at_phase44_frozen_baseline: `True`
- safety_regression: `False`

## Difficult Files

### Test Results 3.pdf

- Final status before Phase 45: `review`
- Final status after Phase 45: `review`
- moved_from_review_ocr_quality_to_review: `False`
- OCR band: `good` score `0.82`
- Selected engine: `tesseract_rus_eng`
- Cyrillic ratio before/after: `0.0 / 0.8561`
- Document type: `microbiology_pcr_report` confidence `0.92`
- Lab table detected: `False`  parsed rows: `0`  coverage band: `none`
- cyrillic_non_lab_document_detected: `False`
- ocr_quality_recovered_non_lab: `False`
- Reason codes: `table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities, non_lab_document_skipped_lab_normalization, microbiology_pcr_report_detected, language_aware_ocr_required`

### Test Results 6.pdf

- Final status before Phase 45: `review_ocr_quality`
- Final status after Phase 45: `review`
- moved_from_review_ocr_quality_to_review: `True`
- OCR band: `good` score `0.82`
- Selected engine: `tesseract_rus_eng`
- Cyrillic ratio before/after: `0.0 / 0.6845`
- Document type: `prescription` confidence `0.95`
- Lab table detected: `False`  parsed rows: `0`  coverage band: `none`
- cyrillic_non_lab_document_detected: `True`
- ocr_quality_recovered_non_lab: `True`
- Reason codes: `table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities, classifier_legacy_ocr_flag, legacy_normalized_low_coverage, non_lab_document_skipped_lab_normalization, document_type_prescription_not_lab, language_aware_ocr_required, cyrillic_non_lab_document_review, ocr_quality_recovered_non_lab, prescription_or_medication_instruction_detected`

## Per-file Results

| File | Before | After | Moved | Doc type | Conf | Band | CyrBefore | CyrAfter | NonLabDetected |
| --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | --- |
| Results 1.pdf | review | review | no | lab_report | 0.69 | usable_with_review | 0 | 0 | no |
| Results 2.pdf | review | review | no | unknown_medical | 0.3 | usable_with_review | 0 | 0 | no |
| Test Results 2.pdf | review_ocr_quality | review_ocr_quality | no | microbiology_pcr_report | 0.92 | good | 0 | 0 | no |
| Test Results 3.pdf | review | review | no | microbiology_pcr_report | 0.92 | good | 0 | 0.8561 | no |
| Test Results 4.pdf | accepted | accepted | no | microbiology_pcr_report | 0.79 | good | 0 | 0 | no |
| Test Results 5.pdf | review_ocr_quality | review_ocr_quality | no | unknown_medical | 0.3 | usable_with_review | 0 | 0 | no |
| Test Results 6.pdf | review_ocr_quality | review | yes | prescription | 0.95 | good | 0 | 0.6845 | yes |
| Urinalysis, Routine.pdf | accepted | accepted | no | lab_report | 0.95 | usable_with_review | 0 | 0 | no |
