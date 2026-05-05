# MedAI Batch Real-Document Validation

- Generated at: `2026-05-01T22:55:16.689422+00:00`
- Input dir: `G:\Codex\2026-04-22-connect-github\holdout_validation_input`
- Total files: `8`
- Accepted: `2`
- Review: `6`
- Errors: `0`
- Empty extractions: `2`
- Fallbacks: `7`
- Low text quality: `0`
- Average text length: `1901.62`

## OCR Quality Summary

- ocr_low_quality_count: `2`
- OCR/Layout routes: `{'scanned_or_low_text': 3, 'digital_clean_text': 5}`
- OCR/Layout quality bands: `{'usable_with_review': 4, 'good': 4}`

## Review Reason Breakdown

- empty_extraction: `2`
- confidence_below_threshold: `6`
- low_entity_count: `3`
- low_coverage: `6`
- low_diversity: `2`
- low_extractor_weight: `4`

## Results

| File | Status | Review type | OCR low quality | Mismatch | Route | Quality | Input score | Engine | Reason codes | Entities | Extractor | Text length | Text method | Suspicious text | Fallback | Confidence | Why reviewed | Review reason | Error |
| --- | --- | --- | --- | --- | --- | --- | ---: | --- | --- | ---: | --- | ---: | --- | --- | --- | ---: | --- | --- | --- |
| Results 1.pdf | review | confidence | no | no | scanned_or_low_text | usable_with_review | 0.745 | existing_pdf_pipeline | low_text_density, table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, lab_report_detected | 2 | phi3 | 1030 | existing_pdf_pipeline | no | phi3 | 0.45 | confidence_below_threshold, low_coverage, low_extractor_weight | confidence_below_reject_threshold |  |
| Results 2.pdf | review | confidence | no | no | scanned_or_low_text | usable_with_review | 0.735 | existing_pdf_pipeline | low_text_density, table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities | 1 | phi3 | 953 | existing_pdf_pipeline | no | phi3 | 0.45 | confidence_below_threshold, low_entity_count, low_coverage, low_extractor_weight | confidence_below_reject_threshold |  |
| Test Results 2.pdf | review_ocr_quality | ocr_quality | yes | good_input_but_downstream_ocr_review | digital_clean_text | good | 0.818 | existing_pdf_pipeline | table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, classifier_legacy_ocr_flag, legacy_normalized_low_coverage | 5 | spacy | 2732 | existing_pdf_pipeline | no |  | 0.63 | confidence_below_threshold, low_coverage | confidence_below_accept_threshold |  |
| Test Results 3.pdf | review | confidence | no | no | digital_clean_text | good | 0.82 | tesseract_rus_eng | table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities, non_lab_document_skipped_lab_normalization, microbiology_pcr_report_detected, language_aware_ocr_required | 0 | phi3 | 3215 | tesseract_rus_eng | no | gemini_quota_or_rate_limit | 0.09 | empty_extraction, confidence_below_threshold, low_entity_count, low_coverage, low_diversity, low_extractor_weight | empty_extraction, confidence_below_reject_threshold |  |
| Test Results 4.pdf | accepted | other | no | no | scanned_or_low_text | good | 0.816 | existing_pdf_pipeline | low_text_density, table_structure_loss, accepted_clean_input | 3 | spacy | 1035 | existing_pdf_pipeline | no | gemini_quota_or_rate_limit | 0.693 |  | accept_with_route_audit |  |
| Test Results 5.pdf | review_ocr_quality | ocr_quality | yes | no | digital_clean_text | usable_with_review | 0.796 | existing_pdf_pipeline | table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, classifier_legacy_ocr_flag, legacy_normalized_low_coverage | 3 | spacy | 2500 | existing_pdf_pipeline | no | gemini_quota_or_rate_limit | 0.63 | confidence_below_threshold, low_coverage | confidence_below_accept_threshold |  |
| Test Results 6.pdf | review | confidence | no | no | digital_clean_text | good | 0.82 | tesseract_rus_eng | table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities, classifier_legacy_ocr_flag, legacy_normalized_low_coverage, non_lab_document_skipped_lab_normalization, document_type_prescription_not_lab, language_aware_ocr_required, cyrillic_non_lab_document_review, ocr_quality_recovered_non_lab, prescription_or_medication_instruction_detected | 0 | phi3 | 1383 | tesseract_rus_eng | no | gemini_quota_or_rate_limit | 0.09 | empty_extraction, confidence_below_threshold, low_entity_count, low_coverage, low_diversity, low_extractor_weight | empty_extraction, confidence_below_reject_threshold |  |
| Urinalysis, Routine.pdf | accepted | other | no | no | digital_clean_text | usable_with_review | 0.925 | existing_pdf_pipeline | accepted_clean_input, lab_report_detected | 11 | phi3 | 2365 | existing_pdf_pipeline | no | phi3 | 0.792 |  | accept_with_route_audit |  |

## Files Needing Review

- `Results 1.pdf`
- `Results 2.pdf`
- `Test Results 2.pdf`
- `Test Results 3.pdf`
- `Test Results 5.pdf`
- `Test Results 6.pdf`

## Errors

- None
