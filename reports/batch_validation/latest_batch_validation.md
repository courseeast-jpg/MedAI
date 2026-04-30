# MedAI Batch Real-Document Validation

- Generated at: `2026-04-30T22:24:19.564291+00:00`
- Input dir: `G:\Codex\2026-04-22-connect-github\real_validation_input`
- Total files: `15`
- Accepted: `9`
- Review: `6`
- Errors: `0`
- Empty extractions: `1`
- Fallbacks: `12`
- Low text quality: `0`
- Average text length: `338.6`

## Review Reason Breakdown

- empty_extraction: `1`
- confidence_below_threshold: `6`
- low_entity_count: `4`
- low_coverage: `1`
- low_diversity: `1`
- low_extractor_weight: `5`

## Results

| File | Status | Entities | Extractor | Text length | Text method | Suspicious text | Fallback | Confidence | Why reviewed | Review reason | Error |
| --- | --- | ---: | --- | ---: | --- | --- | --- | ---: | --- | --- | --- |
| 1.pdf | accepted | 5 | spacy | 408 | tesseract fallback | no |  | 0.83 |  | accept |  |
| 2.pdf | accepted | 13 | spacy | 1485 | tesseract fallback | no |  | 0.838 |  | accept |  |
| 21.pdf | accepted | 4 | spacy | 176 | tesseract fallback | no |  | 0.83 |  | accept |  |
| 22.pdf | review | 0 | spacy | 77 | tesseract fallback | no | spacy | 0.45 | empty_extraction, confidence_below_threshold, low_entity_count, low_coverage, low_diversity | empty_extraction, confidence_below_reject_threshold |  |
| 23.pdf | review | 1 | phi3 | 122 | tesseract fallback | no | phi3 | 0.45 | confidence_below_threshold, low_entity_count, low_extractor_weight | confidence_below_reject_threshold |  |
| 3-1.pdf | accepted | 3 | phi3 | 441 | tesseract fallback | no | phi3 | 0.723 |  | accept_with_route_audit |  |
| 3-2.pdf | review | 2 | phi3 | 302 | tesseract fallback | no | phi3 | 0.45 | confidence_below_threshold, low_extractor_weight | confidence_below_reject_threshold |  |
| 4.pdf | accepted | 3 | phi3 | 645 | tesseract fallback | no | phi3 | 0.723 |  | accept_with_route_audit |  |
| 5.pdf | accepted | 3 | phi3 | 59 | tesseract fallback | no | phi3 | 0.86 |  | accept_with_route_audit |  |
| 6.pdf | accepted | 4 | phi3 | 340 | tesseract fallback | no | phi3 | 0.797 |  | accept_with_route_audit |  |
| 7-1.pdf | accepted | 3 | phi3 | 501 | tesseract fallback | no | phi3 | 0.723 |  | accept_with_route_audit |  |
| 7-2.pdf | accepted | 3 | phi3 | 202 | tesseract fallback | no | phi3 | 0.797 |  | accept_with_route_audit |  |
| 8-1.pdf | review | 1 | phi3 | 52 | tesseract fallback | no | phi3 | 0.45 | confidence_below_threshold, low_entity_count, low_extractor_weight | confidence_below_reject_threshold |  |
| 8-2.pdf | review | 1 | phi3 | 219 | tesseract fallback | no | phi3 | 0.45 | confidence_below_threshold, low_entity_count, low_extractor_weight | confidence_below_reject_threshold |  |
| 9.pdf | review | 2 | phi3 | 50 | tesseract fallback | no | phi3 | 0.45 | confidence_below_threshold, low_extractor_weight | confidence_below_reject_threshold |  |

## Files Needing Review

- `22.pdf`
- `23.pdf`
- `3-2.pdf`
- `8-1.pdf`
- `8-2.pdf`
- `9.pdf`

## Errors

- None
