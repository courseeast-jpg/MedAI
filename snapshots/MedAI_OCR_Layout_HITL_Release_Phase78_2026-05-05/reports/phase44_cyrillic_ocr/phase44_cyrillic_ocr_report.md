# Phase 44 Cyrillic OCR Candidate Generation

- Generated at: `2026-05-01T22:55:06.658839+00:00`

## OCR Capability Report

- Tesseract available: `True`
- Tesseract path: `C:\Program Files\Tesseract-OCR\tesseract.EXE`
- Russian (rus) available: `True`
- English (eng) available: `True`
- Capability warnings: `[]`
- Available languages count: `161`

## Phase 43 Baseline

- Total: `8`
- Accepted: `2`
- review_ocr_quality: `3`

## Phase 44 Result

- Total files: `8`
- Accepted: `2`
- Review: `6`
- review_ocr_quality: `2`
- Empty: `2`
- review_ocr_quality decreased from Phase 43: `True`
- accepted_safe (≤ Phase 43 accepted): `True`
- Cyrillic OCR attempts: `2`
- Cyrillic OCR succeeded: `2`

## Safety Regression Section

- false_accept_on_bad_ocr: `False`
- poor_ocr_auto_accepted: `False`
- accepted_count_increased_without_gate_support: `False`
- cyrillic_ocr_failure_crashed_pipeline: `False`
- safety_regression: `False`

## Difficult Files

### Test Results 3.pdf

- Final status: `review`
- Document type: `microbiology_pcr_report`  language hint: `ru`
- Language-aware OCR recommended: `True`
- Candidates attempted: `['existing_pdf_pipeline', 'pymupdf_native_text', 'tesseract_rus_eng']`
- Selected engine: `tesseract_rus_eng`  band: `good`  score: `0.82`
- Cyrillic OCR attempted: `True`  engine: `tesseract_rus_eng`  text length: `3215`
- Cyrillic OCR warnings: `['low_medical_token_density']`
- Cyrillic ratio before/after: `0.0 / 0.8561`
- Reason codes: `table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities, non_lab_document_skipped_lab_normalization, microbiology_pcr_report_detected, language_aware_ocr_required`

### Test Results 6.pdf

- Final status: `review`
- Document type: `prescription`  language hint: `ru`
- Language-aware OCR recommended: `True`
- Candidates attempted: `['existing_pdf_pipeline', 'pymupdf_native_text', 'tesseract_rus_eng']`
- Selected engine: `tesseract_rus_eng`  band: `good`  score: `0.82`
- Cyrillic OCR attempted: `True`  engine: `tesseract_rus_eng`  text length: `1383`
- Cyrillic OCR warnings: `['layout_or_table_heavy', 'low_medical_token_density']`
- Cyrillic ratio before/after: `0.0 / 0.6845`
- Reason codes: `table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities, classifier_legacy_ocr_flag, legacy_normalized_low_coverage, non_lab_document_skipped_lab_normalization, document_type_prescription_not_lab, language_aware_ocr_required, cyrillic_non_lab_document_review, ocr_quality_recovered_non_lab, prescription_or_medication_instruction_detected`

## Per-file Results

| File | Status | Doc type | Lang | Selected | Score | CyrOCR | CyrText | CyrFail |
| --- | --- | --- | --- | --- | ---: | --- | ---: | --- |
| Results 1.pdf | review | lab_report | en | existing_pdf_pipeline | 0.745 | no | 0 | no |
| Results 2.pdf | review | unknown_medical | en | existing_pdf_pipeline | 0.735 | no | 0 | no |
| Test Results 2.pdf | review_ocr_quality | microbiology_pcr_report | en | existing_pdf_pipeline | 0.818 | no | 0 | no |
| Test Results 3.pdf | review | microbiology_pcr_report | ru | tesseract_rus_eng | 0.82 | yes | 3215 | no |
| Test Results 4.pdf | accepted | microbiology_pcr_report | en | existing_pdf_pipeline | 0.816 | no | 0 | no |
| Test Results 5.pdf | review_ocr_quality | unknown_medical | en | existing_pdf_pipeline | 0.796 | no | 0 | no |
| Test Results 6.pdf | review | prescription | ru | tesseract_rus_eng | 0.82 | yes | 1383 | no |
| Urinalysis, Routine.pdf | accepted | lab_report | en | existing_pdf_pipeline | 0.925 | no | 0 | no |
