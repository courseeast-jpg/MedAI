# Phase 40 Lab Table Normalization

- Generated at: `2026-05-01T22:18:39.156139+00:00`

## Baselines

- Phase37: `{'total_files': 8, 'accepted': 2, 'review_ocr_quality': 6, 'empty': 0}`
- Phase38: `{'total_files': 8, 'accepted': 2, 'review': 6, 'review_ocr_quality': 4, 'empty': 0}`
- Phase39: `{'total_files': 8, 'ocr_status_mismatches': 3, 'review_ocr_quality': 4, 'safety_regression': False}`

## Phase40 Result

- Total files: `8`
- Accepted: `2`
- Review: `6`
- review_ocr_quality: `2`
- Empty: `2`
- review_ocr_quality decreased from Phase39: `True`

## Safety

- false_accept_on_poor_ocr: `False`
- accepted_due_to_lab_normalizer: `False`
- empty_extraction_leakage: `False`
- phase37_gate_bypassed: `False`

## Per-file Results

| File | OCR band | Before | After | Lab table | Rows | Coverage | Band | Recovered | Reason codes |
| --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| Results 1.pdf | usable_with_review | review | review | yes | 0 | 0.0 | none | no | low_text_density, table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, lab_report_detected |
| Results 2.pdf | usable_with_review | review | review | yes | 0 | 0.0 | none | no | low_text_density, table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities |
| Test Results 2.pdf | good | review_ocr_quality | review_ocr_quality | yes | 9 | 0.077 | weak | no | table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, classifier_legacy_ocr_flag, legacy_normalized_low_coverage |
| Test Results 3.pdf | good | review | review | no | 0 | 0.0 | none | no | table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities, non_lab_document_skipped_lab_normalization, microbiology_pcr_report_detected, language_aware_ocr_required |
| Test Results 4.pdf | good | accepted | accepted | yes | 1 | 0.024 | weak | no | low_text_density, table_structure_loss, accepted_clean_input |
| Test Results 5.pdf | usable_with_review | review_ocr_quality | review_ocr_quality | yes | 3 | 0.041 | weak | no | table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, classifier_legacy_ocr_flag, legacy_normalized_low_coverage |
| Test Results 6.pdf | good | review_ocr_quality | review | no | 0 | 0.0 | none | no | table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities, classifier_legacy_ocr_flag, legacy_normalized_low_coverage, non_lab_document_skipped_lab_normalization, document_type_prescription_not_lab, language_aware_ocr_required, cyrillic_non_lab_document_review, ocr_quality_recovered_non_lab, prescription_or_medication_instruction_detected |
| Urinalysis, Routine.pdf | usable_with_review | accepted | accepted | yes | 14 | 0.167 | partial | no | accepted_clean_input, lab_report_detected |
