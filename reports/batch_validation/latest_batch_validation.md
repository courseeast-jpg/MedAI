# MedAI Batch Real-Document Validation

- Generated at: `2026-04-30T19:07:43.404357+00:00`
- Input dir: `G:\Codex\2026-04-22-connect-github\real_validation_input`
- Total files: `15`
- Accepted: `3`
- Review: `12`
- Errors: `0`
- Empty extractions: `3`
- Fallbacks: `6`
- Low text quality: `0`
- Average text length: `338.6`

## Results

| File | Status | Entities | Extractor | Text length | Text method | Suspicious text | Fallback | Confidence | Review reason | Error |
| --- | --- | ---: | --- | ---: | --- | --- | --- | ---: | --- | --- |
| 1.pdf | review | 7 | gemini | 408 | tesseract fallback | no | confidence_too_low:0.120 | 0.8 | accept_with_route_audit |  |
| 2.pdf | review | 1 | spacy | 1485 | tesseract fallback | no |  | 0.425 | confidence_below_reject_threshold |  |
| 21.pdf | accepted | 4 | spacy | 176 | tesseract fallback | no | confidence_too_low:0.090 | 0.83 | accept_with_route_audit |  |
| 22.pdf | review | 0 | spacy | 77 | tesseract fallback | no | spacy | 0.45 | empty_extraction, confidence_below_reject_threshold |  |
| 23.pdf | review | 1 | phi3 | 122 | tesseract fallback | no | phi3 | 0.45 | confidence_below_reject_threshold |  |
| 3-1.pdf | review | 1 | spacy | 441 | tesseract fallback | no |  | 0.487 | confidence_below_reject_threshold |  |
| 3-2.pdf | review | 2 | spacy | 302 | tesseract fallback | no |  | 0.562 | confidence_below_accept_threshold |  |
| 4.pdf | accepted | 4 | spacy | 645 | tesseract fallback | no |  | 0.693 | accept |  |
| 5.pdf | review | 1 | spacy | 59 | tesseract fallback | no |  | 0.562 | confidence_below_accept_threshold |  |
| 6.pdf | review | 2 | spacy | 340 | tesseract fallback | no |  | 0.637 | confidence_below_accept_threshold |  |
| 7-1.pdf | review | 2 | spacy | 501 | tesseract fallback | no |  | 0.562 | confidence_below_accept_threshold |  |
| 7-2.pdf | accepted | 2 | spacy | 202 | tesseract fallback | no |  | 0.7 | accept |  |
| 8-1.pdf | review | 0 | spacy | 52 | tesseract fallback | no | spacy | 0.45 | empty_extraction, confidence_below_reject_threshold |  |
| 8-2.pdf | review | 1 | spacy | 219 | tesseract fallback | no |  | 0.487 | confidence_below_reject_threshold |  |
| 9.pdf | review | 0 | spacy | 50 | tesseract fallback | no | spacy | 0.45 | empty_extraction, confidence_below_reject_threshold |  |

## Files Needing Review

- `1.pdf`
- `2.pdf`
- `22.pdf`
- `23.pdf`
- `3-1.pdf`
- `3-2.pdf`
- `5.pdf`
- `6.pdf`
- `7-1.pdf`
- `8-1.pdf`
- `8-2.pdf`
- `9.pdf`

## Errors

- None
