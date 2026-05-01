# MedAI Batch Real-Document Validation

- Generated at: `2026-05-01T00:06:21.606781+00:00`
- Input dir: `G:\Codex\2026-04-22-connect-github\holdout_validation_input`
- Total files: `8`
- Accepted: `4`
- Review: `4`
- Errors: `0`
- Empty extractions: `0`
- Fallbacks: `8`
- Low text quality: `2`
- Average text length: `2040.25`

## Review Reason Breakdown

- empty_extraction: `0`
- confidence_below_threshold: `4`
- low_entity_count: `1`
- low_coverage: `4`
- low_diversity: `0`
- low_extractor_weight: `1`

## Results

| File | Status | Entities | Extractor | Text length | Text method | Suspicious text | Fallback | Confidence | Why reviewed | Review reason | Error |
| --- | --- | ---: | --- | ---: | --- | --- | --- | ---: | --- | --- | --- |
| Results 1.pdf | accepted | 4 | spacy | 1009 | tesseract fallback | yes | latency_too_high:13036.000 | 0.767 |  | accept_with_route_audit |  |
| Results 2.pdf | review | 1 | phi3 | 939 | tesseract fallback | yes | phi3 | 0.45 | confidence_below_threshold, low_entity_count, low_coverage, low_extractor_weight | confidence_below_reject_threshold |  |
| Test Results 2.pdf | accepted | 6 | spacy | 2689 | unknown | no | gemini_quota_or_rate_limit | 0.7 |  | accept_with_route_audit |  |
| Test Results 3.pdf | review | 4 | spacy | 3723 | unknown | no | gemini_quota_or_rate_limit | 0.63 | confidence_below_threshold, low_coverage | confidence_below_accept_threshold |  |
| Test Results 4.pdf | accepted | 4 | spacy | 997 | tesseract fallback | no | gemini_quota_or_rate_limit | 0.693 |  | accept_with_route_audit |  |
| Test Results 5.pdf | review | 3 | spacy | 2312 | unknown | no | gemini_quota_or_rate_limit | 0.63 | confidence_below_threshold, low_coverage | confidence_below_accept_threshold |  |
| Test Results 6.pdf | review | 3 | spacy | 2325 | unknown | no | gemini_quota_or_rate_limit | 0.63 | confidence_below_threshold, low_coverage | confidence_below_accept_threshold |  |
| Urinalysis, Routine.pdf | accepted | 12 | phi3 | 2328 | unknown | no | phi3 | 0.792 |  | accept_with_route_audit |  |

## Files Needing Review

- `Results 2.pdf`
- `Test Results 3.pdf`
- `Test Results 5.pdf`
- `Test Results 6.pdf`

## Errors

- None
