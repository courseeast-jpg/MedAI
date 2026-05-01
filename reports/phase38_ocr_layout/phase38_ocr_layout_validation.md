# Phase 38 OCR/Layout Validation

- Generated at: `2026-05-01T15:26:49.390079+00:00`
- Input dir: `G:\Codex\2026-04-22-connect-github\holdout_validation_input`

## Summary

- Total files: `8`
- Accepted: `2`
- Review: `6`
- review_ocr_quality: `4`
- Empty: `0`
- OCR/Layout routes: `{'scanned_or_low_text': 3, 'digital_clean_text': 5}`
- OCR/Layout quality bands: `{'usable_with_review': 4, 'good': 4}`

## Phase 37 Comparison

- Phase37 holdout total: `8`
- Phase37 accepted: `2`
- Phase37 review_ocr_quality: `6`
- Phase37 empty: `0`
- Phase37 review_ocr_quality improved: `2`
- OCR/Layout improved input quality: `True`
- Safety regression: `False`

## Per-file OCR/Layout Decisions

| File | Status | Route | Quality | Score | Selected engine | Phase37 OCR review improved |
| --- | --- | --- | --- | ---: | --- | --- |
| Results 1.pdf | review | scanned_or_low_text | usable_with_review | 0.745 | existing_pdf_pipeline | yes |
| Results 2.pdf | review | scanned_or_low_text | usable_with_review | 0.735 | existing_pdf_pipeline | yes |
| Test Results 2.pdf | review_ocr_quality | digital_clean_text | good | 0.818 | existing_pdf_pipeline | no |
| Test Results 3.pdf | review_ocr_quality | digital_clean_text | good | 0.811 | existing_pdf_pipeline | no |
| Test Results 4.pdf | accepted | scanned_or_low_text | good | 0.816 | existing_pdf_pipeline | no |
| Test Results 5.pdf | review_ocr_quality | digital_clean_text | usable_with_review | 0.796 | existing_pdf_pipeline | no |
| Test Results 6.pdf | review_ocr_quality | digital_clean_text | good | 0.809 | existing_pdf_pipeline | no |
| Urinalysis, Routine.pdf | accepted | digital_clean_text | usable_with_review | 0.925 | existing_pdf_pipeline | no |
