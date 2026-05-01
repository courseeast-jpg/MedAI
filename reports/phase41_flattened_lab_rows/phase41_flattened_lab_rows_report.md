# Phase 41 Flattened Lab Row Parser Expansion

- Generated at: `2026-05-01T18:59:54.816589+00:00`

## Phase 40 Baseline

- Total: `8`
- Accepted: `2`
- review_ocr_quality: `3`

## Phase 41 Result

- Total files: `8`
- Accepted: `2`
- Review: `6`
- review_ocr_quality: `3`
- Empty: `0`
- review_ocr_quality vs Phase 40: `+0`
- review_ocr_quality decreased from Phase 40: `False`
- accepted_safe (≤ Phase40 accepted): `True`

## Safety

- false_accept_on_poor_ocr: `False`
- accepted_due_to_lab_normalizer: `False`
- empty_extraction_leakage: `False`
- safety_regression: `False`

## Difficult Files

### Test Results 3.pdf

- Status: `review_ocr_quality`
- Parsed rows: `1`
- Flattened rows recovered: `0`
- Coverage ratio: `0.018`
- Coverage band: `weak`
- Reason codes: `table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, classifier_legacy_ocr_flag, legacy_normalized_low_coverage`

### Test Results 6.pdf

- Status: `review`
- Parsed rows: `0`
- Flattened rows recovered: `0`
- Coverage ratio: `0.0`
- Coverage band: `none`
- Reason codes: `table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities`

## Per-file Results

| File | OCR band | Status | Rows | Flattened | Coverage | Band | Upgraded | Reason codes |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| Results 1.pdf | usable_with_review | review | 0 | 0 | 0.0 | none | no | low_text_density, table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence |
| Results 2.pdf | usable_with_review | review | 0 | 0 | 0.0 | none | no | low_text_density, table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities |
| Test Results 2.pdf | good | review_ocr_quality | 9 | 0 | 0.077 | weak | no | table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, classifier_legacy_ocr_flag, legacy_normalized_low_coverage |
| Test Results 3.pdf | good | review_ocr_quality | 1 | 0 | 0.018 | weak | no | table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, classifier_legacy_ocr_flag, legacy_normalized_low_coverage |
| Test Results 4.pdf | good | accepted | 1 | 0 | 0.024 | weak | no | low_text_density, table_structure_loss, accepted_clean_input |
| Test Results 5.pdf | usable_with_review | review_ocr_quality | 3 | 0 | 0.041 | weak | no | table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, classifier_legacy_ocr_flag, legacy_normalized_low_coverage |
| Test Results 6.pdf | good | review | 0 | 0 | 0.0 | none | no | table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities |
| Urinalysis, Routine.pdf | usable_with_review | accepted | 14 | 2 | 0.167 | partial | no | accepted_clean_input |
