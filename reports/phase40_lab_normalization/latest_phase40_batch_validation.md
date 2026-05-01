# MedAI Batch Real-Document Validation

- Generated at: `2026-05-01T16:07:54.317983+00:00`
- Input dir: `G:\Codex\2026-04-22-connect-github\holdout_validation_input`
- Total files: `8`
- Accepted: `2`
- Review: `6`
- Errors: `0`
- Empty extractions: `0`
- Fallbacks: `7`
- Low text quality: `0`
- Average text length: `2102.5`

## OCR Quality Summary

- ocr_low_quality_count: `3`
- OCR/Layout routes: `{'scanned_or_low_text': 3, 'digital_clean_text': 5}`
- OCR/Layout quality bands: `{'usable_with_review': 4, 'good': 4}`

## Review Reason Breakdown

- empty_extraction: `0`
- confidence_below_threshold: `5`
- low_entity_count: `1`
- low_coverage: `6`
- low_diversity: `0`
- low_extractor_weight: `2`

## Results

| File | Status | Review type | OCR low quality | Mismatch | Route | Quality | Input score | Engine | Reason codes | Entities | Extractor | Text length | Text method | Suspicious text | Fallback | Confidence | Why reviewed | Review reason | Error |
| --- | --- | --- | --- | --- | --- | --- | ---: | --- | --- | ---: | --- | ---: | --- | --- | --- | ---: | --- | --- | --- |
| Results 1.pdf | review | confidence | no | no | scanned_or_low_text | usable_with_review | 0.745 | existing_pdf_pipeline | low_text_density, table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence | 2 | phi3 | 1030 | existing_pdf_pipeline | no | phi3 | 0.45 | confidence_below_threshold, low_coverage, low_extractor_weight | confidence_below_reject_threshold |  |
| Results 2.pdf | review | confidence | no | no | scanned_or_low_text | usable_with_review | 0.735 | existing_pdf_pipeline | low_text_density, table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities | 1 | phi3 | 953 | existing_pdf_pipeline | no | phi3 | 0.45 | confidence_below_threshold, low_entity_count, low_coverage, low_extractor_weight | confidence_below_reject_threshold |  |
| Test Results 2.pdf | review | confidence | no | no | digital_clean_text | good | 0.818 | existing_pdf_pipeline | table_structure_loss, extraction_low_coverage, lab_table_recovered, lab_table_recovered_review_only | 6 | spacy | 2732 | existing_pdf_pipeline | no |  | 0.7 | low_coverage | accept |  |
| Test Results 3.pdf | review_ocr_quality | ocr_quality | yes | good_input_but_downstream_ocr_review | digital_clean_text | good | 0.811 | existing_pdf_pipeline | table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, classifier_legacy_ocr_flag, legacy_normalized_low_coverage | 4 | spacy | 3900 | existing_pdf_pipeline | no | gemini_quota_or_rate_limit | 0.63 | confidence_below_threshold, low_coverage | confidence_below_accept_threshold |  |
| Test Results 4.pdf | accepted | other | no | no | scanned_or_low_text | good | 0.816 | existing_pdf_pipeline | low_text_density, table_structure_loss, accepted_clean_input | 4 | spacy | 1035 | existing_pdf_pipeline | no | gemini_quota_or_rate_limit | 0.693 |  | accept_with_route_audit |  |
| Test Results 5.pdf | review_ocr_quality | ocr_quality | yes | no | digital_clean_text | usable_with_review | 0.796 | existing_pdf_pipeline | table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, classifier_legacy_ocr_flag, legacy_normalized_low_coverage | 3 | spacy | 2500 | existing_pdf_pipeline | no | gemini_quota_or_rate_limit | 0.63 | confidence_below_threshold, low_coverage | confidence_below_accept_threshold |  |
| Test Results 6.pdf | review_ocr_quality | ocr_quality | yes | good_input_but_downstream_ocr_review | digital_clean_text | good | 0.809 | existing_pdf_pipeline | table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, classifier_legacy_ocr_flag, legacy_normalized_low_coverage | 3 | spacy | 2305 | existing_pdf_pipeline | no | gemini_quota_or_rate_limit | 0.63 | confidence_below_threshold, low_coverage | confidence_below_accept_threshold |  |
| Urinalysis, Routine.pdf | accepted | other | no | no | digital_clean_text | usable_with_review | 0.925 | existing_pdf_pipeline | accepted_clean_input | 12 | phi3 | 2365 | existing_pdf_pipeline | no | phi3 | 0.792 |  | accept_with_route_audit |  |

## Files Needing Review

- `Results 1.pdf`
- `Results 2.pdf`
- `Test Results 2.pdf`
- `Test Results 3.pdf`
- `Test Results 5.pdf`
- `Test Results 6.pdf`

## Errors

- None
