# MEDAI-RU-LAB-EXTRACT-DIAG-01

Conclusion: medai_ru_lab_extract_diag_01_completed
Diagnostic type: russian_lab_pdf_text_visibility
Files analyzed: 2

## Root Cause Candidates

1. no_cyrillic_text_detected
2. text_extraction_visible_but_lab_cues_absent
3. text_too_sparse_or_table_structure_lost

Likely primary cause: no_cyrillic_text_detected

## Safe Per-File Diagnostic Summary

| Safe ID | Current type | Extractor | Confidence | OCR quality | Text | Cyrillic | Digits | Table-like | Lines | Cue count | Cue categories | Likely reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| file_001 | Unknown | spacy | moderate_0_50_to_0_64 | readable | long | none | medium | True | many | 0 | none | no_cyrillic_text_detected |
| file_002 | Unknown | rules_based | low_under_0_50 | readable | medium | none | medium | True | many | 0 | none | no_cyrillic_text_detected |

## Recommendation

Recommended next block: MEDAI-RU-LAB-TEXT-VIS-01 - Russian lab extraction text visibility repair

Deferred actions:
- External OCR: not recommended yet
- Threshold changes: not recommended
- Auto-accept expansion: not recommended
- Parser rewrite: wait until text visibility is understood

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
