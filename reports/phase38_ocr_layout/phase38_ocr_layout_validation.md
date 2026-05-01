# Phase 38 OCR/Layout Validation

- Generated at: `2026-05-01T22:18:48.395630+00:00`
- Input dir: `G:\Codex\2026-04-22-connect-github\holdout_validation_input`

## Summary

- Total files: `8`
- Accepted: `3`
- Review: `5`
- review_ocr_quality: `2`
- Empty: `2`
- OCR/Layout routes: `{'scanned_or_low_text': 3, 'digital_clean_text': 5}`
- OCR/Layout quality bands: `{'usable_with_review': 4, 'good': 4}`

## Phase 37 Comparison

- Phase37 holdout total: `8`
- Phase37 accepted: `2`
- Phase37 review_ocr_quality: `6`
- Phase37 empty: `0`
- Phase37 review_ocr_quality improved: `4`
- OCR status mismatches: `1`
- OCR/Layout improved input quality: `True`
- Safety regression: `False`

## Per-file OCR/Layout Decisions

| File | Status | Route | Quality | Score | Selected engine | Mismatch | Reason codes | Phase37 OCR review improved |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- |
| Results 1.pdf | accepted | scanned_or_low_text | usable_with_review | 0.745 | existing_pdf_pipeline | no | low_text_density, table_structure_loss, accepted_clean_input, lab_report_detected | yes |
| Results 2.pdf | review | scanned_or_low_text | usable_with_review | 0.735 | existing_pdf_pipeline | no | low_text_density, table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities | yes |
| Test Results 2.pdf | review_ocr_quality | digital_clean_text | good | 0.818 | existing_pdf_pipeline | good_input_but_downstream_ocr_review | table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, classifier_legacy_ocr_flag, legacy_normalized_low_coverage | no |
| Test Results 3.pdf | review | digital_clean_text | good | 0.82 | tesseract_rus_eng | no | table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities, non_lab_document_skipped_lab_normalization, microbiology_pcr_report_detected, language_aware_ocr_required | yes |
| Test Results 4.pdf | accepted | scanned_or_low_text | good | 0.816 | existing_pdf_pipeline | no | low_text_density, table_structure_loss, accepted_clean_input | no |
| Test Results 5.pdf | review_ocr_quality | digital_clean_text | usable_with_review | 0.796 | existing_pdf_pipeline | no | table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, classifier_legacy_ocr_flag, legacy_normalized_low_coverage | no |
| Test Results 6.pdf | review | digital_clean_text | good | 0.82 | tesseract_rus_eng | no | table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities, classifier_legacy_ocr_flag, legacy_normalized_low_coverage, non_lab_document_skipped_lab_normalization, document_type_prescription_not_lab, language_aware_ocr_required, cyrillic_non_lab_document_review, ocr_quality_recovered_non_lab, prescription_or_medication_instruction_detected | yes |
| Urinalysis, Routine.pdf | accepted | digital_clean_text | usable_with_review | 0.925 | existing_pdf_pipeline | no | accepted_clean_input, lab_report_detected | no |
