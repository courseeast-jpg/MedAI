# MEDAI-RU-LAB-TEXT-VIS-01

Conclusion: medai_ru_lab_text_vis_01_completed
Diagnostic type: russian_pdf_cyrillic_text_visibility
Files analyzed: 2
Cyrillic text visibility issue confirmed: true
Numeric table text visible: true
OCR routing change recommended: true

## Root Cause Candidates

1. ocr_skipped_due_to_numeric_readable_text
2. embedded_text_has_no_cyrillic
3. classifier_receives_numeric_only_text
4. tesseract_russian_language_missing

Likely primary cause: ocr_skipped_due_to_numeric_readable_text

## Safe Per-File Diagnostic Summary

| Safe ID | Extractor path | PDF text | OCR attempted | OCR engine | OCR language | Text | Cyrillic | Digits | ASCII letters | Table-like | Non-ASCII | Likely reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| file_001 | pymupdf_native_text | True | False | none | multilingual | long | none | medium | high | True | True | ocr_skipped_due_to_numeric_readable_text |
| file_002 | pymupdf_native_text | True | False | none | multilingual | long | none | medium | high | True | True | ocr_skipped_due_to_numeric_readable_text |

## Recommendation

Recommended next block: MEDAI-RU-LAB-OCR-GATE-01 - Cyrillic visibility OCR gate diagnostic

Deferred actions:
- External OCR: not recommended
- Auto-accept expansion: not recommended
- Confidence threshold changes: not recommended
- Parser rewrite: wait until Cyrillic text visibility is understood

## Safety

- Clinical logic changed: false
- Clinical interpretation added: false
- Medication advice added: false
- DDI logic changed: false
- Confidence thresholds changed: false
- Auto-acceptance changed: false
- OCR engine changed: false
- External API enabled: false
- Cloud API used: false
- Extraction parser changed: false
- Lab value parser added: false
- Treatment parser added: false
- Safety gate changed: false
- B07 terminology changed: false
- ROUTE-FIX changed: false
- DB schema changed: false
- Command behavior changed: false
- Allowlist changed: false

## Privacy

- No raw PHI in report: true
- No raw filenames in report: true
- No raw document text in report: true
- No private paths in report: true
- No secrets in report: true
