# MedAI Batch Real-Document Validation

- Generated at: `2026-05-01T15:25:27.675170+00:00`
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

- ocr_low_quality_count: `4`
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

| File | Status | Review type | OCR low quality | Route | Quality | Input score | Engine | Entities | Extractor | Text length | Text method | Suspicious text | Fallback | Confidence | Why reviewed | Review reason | Error |
| --- | --- | --- | --- | --- | --- | ---: | --- | ---: | --- | ---: | --- | --- | --- | ---: | --- | --- | --- |
| Results 1.pdf | review | confidence | no | scanned_or_low_text | usable_with_review | 0.745 | existing_pdf_pipeline | 2 | phi3 | 1030 | existing_pdf_pipeline | no | phi3 | 0.45 | confidence_below_threshold, low_coverage, low_extractor_weight | confidence_below_reject_threshold |  |
| Results 2.pdf | review | confidence | no | scanned_or_low_text | usable_with_review | 0.735 | existing_pdf_pipeline | 1 | phi3 | 953 | existing_pdf_pipeline | no | phi3 | 0.45 | confidence_below_threshold, low_entity_count, low_coverage, low_extractor_weight | confidence_below_reject_threshold |  |
| Test Results 2.pdf | review_ocr_quality | ocr_quality | yes | digital_clean_text | good | 0.818 | existing_pdf_pipeline | 6 | spacy | 2732 | existing_pdf_pipeline | no |  | 0.7 | low_coverage | accept |  |
| Test Results 3.pdf | review_ocr_quality | ocr_quality | yes | digital_clean_text | good | 0.811 | existing_pdf_pipeline | 4 | spacy | 3900 | existing_pdf_pipeline | no | gemini_quota_or_rate_limit | 0.63 | confidence_below_threshold, low_coverage | confidence_below_accept_threshold |  |
| Test Results 4.pdf | accepted | other | no | scanned_or_low_text | good | 0.816 | existing_pdf_pipeline | 4 | spacy | 1035 | existing_pdf_pipeline | no | gemini_quota_or_rate_limit | 0.693 |  | accept_with_route_audit |  |
| Test Results 5.pdf | review_ocr_quality | ocr_quality | yes | digital_clean_text | usable_with_review | 0.796 | existing_pdf_pipeline | 3 | spacy | 2500 | existing_pdf_pipeline | no | gemini_quota_or_rate_limit | 0.63 | confidence_below_threshold, low_coverage | confidence_below_accept_threshold |  |
| Test Results 6.pdf | review_ocr_quality | ocr_quality | yes | digital_clean_text | good | 0.809 | existing_pdf_pipeline | 3 | spacy | 2305 | existing_pdf_pipeline | no | gemini_quota_or_rate_limit | 0.63 | confidence_below_threshold, low_coverage | confidence_below_accept_threshold |  |
| Urinalysis, Routine.pdf | accepted | other | no | digital_clean_text | usable_with_review | 0.925 | existing_pdf_pipeline | 12 | phi3 | 2365 | existing_pdf_pipeline | no | phi3 | 0.792 |  | accept_with_route_audit |  |

## Files Needing Review

- `Results 1.pdf`
- `Results 2.pdf`
- `Test Results 2.pdf`
- `Test Results 3.pdf`
- `Test Results 5.pdf`
- `Test Results 6.pdf`

## Errors

- None
