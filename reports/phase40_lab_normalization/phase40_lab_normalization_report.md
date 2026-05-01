# Phase 40 Lab Table Normalization

- Generated at: `2026-05-01T16:08:48.654594+00:00`

## Baselines

- Phase37: `{'total_files': 8, 'accepted': 2, 'review_ocr_quality': 6, 'empty': 0}`
- Phase38: `{'total_files': 8, 'accepted': 2, 'review': 6, 'review_ocr_quality': 4, 'empty': 0}`
- Phase39: `{'total_files': 8, 'ocr_status_mismatches': 3, 'review_ocr_quality': 4, 'safety_regression': False}`

## Phase40 Result

- Total files: `8`
- Accepted: `2`
- Review: `6`
- review_ocr_quality: `3`
- Empty: `0`
- review_ocr_quality decreased from Phase39: `True`

## Safety

- false_accept_on_poor_ocr: `False`
- accepted_due_to_lab_normalizer: `False`
- empty_extraction_leakage: `False`
- phase37_gate_bypassed: `False`

## Per-file Results

| File | OCR band | Before | After | Lab table | Rows | Coverage | Band | Recovered | Reason codes |
| --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| Results 1.pdf | usable_with_review | review | review | yes | 0 | 0.0 | none | no | low_text_density, table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence |
| Results 2.pdf | usable_with_review | review | review | yes | 0 | 0.0 | none | no | low_text_density, table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities |
| Test Results 2.pdf | good | review_ocr_quality | review | yes | 9 | 0.085 | partial | yes | table_structure_loss, extraction_low_coverage, lab_table_recovered, lab_table_recovered_review_only |
| Test Results 3.pdf | good | review_ocr_quality | review_ocr_quality | yes | 1 | 0.02 | weak | no | table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, classifier_legacy_ocr_flag, legacy_normalized_low_coverage |
| Test Results 4.pdf | good | accepted | accepted | yes | 1 | 0.025 | weak | no | low_text_density, table_structure_loss, accepted_clean_input |
| Test Results 5.pdf | usable_with_review | review_ocr_quality | review_ocr_quality | yes | 3 | 0.043 | weak | no | table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, classifier_legacy_ocr_flag, legacy_normalized_low_coverage |
| Test Results 6.pdf | good | review_ocr_quality | review_ocr_quality | yes | 0 | 0.0 | none | no | table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, classifier_legacy_ocr_flag, legacy_normalized_low_coverage |
| Urinalysis, Routine.pdf | usable_with_review | accepted | accepted | yes | 14 | 0.171 | partial | no | accepted_clean_input |
