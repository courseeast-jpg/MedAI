# Phase 43 Document Type and Language-Aware Routing

- Generated at: `2026-05-01T19:17:30.384868+00:00`

## Phase 42 Baseline

- Total: `8`
- Accepted: `2`
- review_ocr_quality: `3`

## Phase 43 Result

- Total files: `8`
- Accepted: `2`
- Review: `6`
- review_ocr_quality: `3`
- Empty: `0`
- review_ocr_quality decreased from Phase 42: `False`
- accepted_safe (≤ Phase 42 accepted): `True`
- Document type distribution: `{'lab_report': 2, 'unknown_medical': 2, 'microbiology_pcr_report': 3, 'prescription': 1}`

## Safety Regression Section

- false_accept_on_bad_ocr: `False`
- poor_ocr_auto_accepted: `False`
- lab_parser_bypassed_unsafely: `False`
- accepted_count_increased_without_gate_support: `False`
- safety_regression: `False`

## Difficult Files

### Test Results 3.pdf

- Final status: `review_ocr_quality`
- Document type: `microbiology_pcr_report` (confidence `0.92`)
- Language hint: `mixed`
- should_run_lab_normalization: `False`
- should_recommend_language_aware_ocr: `True`
- Lab parser skipped: `True`
- Reason codes: `table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, classifier_legacy_ocr_flag, legacy_normalized_low_coverage, non_lab_document_skipped_lab_normalization, microbiology_pcr_report_detected, language_aware_ocr_required`

### Test Results 6.pdf

- Final status: `review`
- Document type: `prescription` (confidence `0.95`)
- Language hint: `mixed`
- should_run_lab_normalization: `False`
- should_recommend_language_aware_ocr: `True`
- Lab parser skipped: `True`
- Reason codes: `table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities, non_lab_document_skipped_lab_normalization, document_type_prescription_not_lab, language_aware_ocr_required`

## Per-file Results

| File | OCR band | Status | Doc type | Conf | Lang | Skipped | LangOCR rec | Reason codes |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- |
| Results 1.pdf | usable_with_review | review | lab_report | 0.69 | en | no | no | low_text_density, table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, lab_report_detected |
| Results 2.pdf | usable_with_review | review | unknown_medical | 0.3 | en | no | no | low_text_density, table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities |
| Test Results 2.pdf | good | review_ocr_quality | microbiology_pcr_report | 0.92 | en | no | no | table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, classifier_legacy_ocr_flag, legacy_normalized_low_coverage |
| Test Results 3.pdf | good | review_ocr_quality | microbiology_pcr_report | 0.92 | mixed | yes | yes | table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, classifier_legacy_ocr_flag, legacy_normalized_low_coverage, non_lab_document_skipped_lab_normalization, microbiology_pcr_report_detected, language_aware_ocr_required |
| Test Results 4.pdf | good | accepted | microbiology_pcr_report | 0.79 | en | no | no | low_text_density, table_structure_loss, accepted_clean_input |
| Test Results 5.pdf | usable_with_review | review_ocr_quality | unknown_medical | 0.3 | en | no | no | table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, classifier_legacy_ocr_flag, legacy_normalized_low_coverage |
| Test Results 6.pdf | good | review | prescription | 0.95 | mixed | yes | yes | table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities, non_lab_document_skipped_lab_normalization, document_type_prescription_not_lab, language_aware_ocr_required |
| Urinalysis, Routine.pdf | usable_with_review | accepted | lab_report | 0.95 | en | no | no | accepted_clean_input, lab_report_detected |
