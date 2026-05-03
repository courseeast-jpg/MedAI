# Phase53 Blind PDF Generalization Audit

- Timestamp: `2026-05-03T03:43:46.876456+00:00`
- Total files: `15`
- Processed files: `15`
- Accepted: `12`
- Review: `3`
- Review OCR quality: `1`
- Empty: `2`
- Errors: `0`
- Local-only mode: `True`
- External API default allowed: `False`
- Privacy gate status: `local_only`
- Raw PHI logged: `False`
- Raw PHI logged in public reports: `False`
- Report PDF artifacts tracked: `False`
- Conclusion: `PASS_SAFETY_ACCEPTABLE_AUTOMATION`

## Per-File Results

| Safe File ID | Filename Hash | Content Hash | Type | Status | Outcome | Extractor | Confidence | OCR status | Reason codes | Error |
| --- | --- | ---: | --- | --- | --- | ---: | --- | --- | --- |
| `file_001` | `e11cbfc531a99905` | `446fb63dd542670a` | `.pdf` | `accepted` | `written` | `spacy` | `0.9` | `good` | `low_text_density, accepted_clean_input` | `` |
| `file_002` | `111dae5cbe3d92db` | `f762ca21a7b77a8a` | `.pdf` | `accepted` | `written` | `spacy` | `0.838` | `good` | `low_text_density, accepted_clean_input, lab_report_detected` | `` |
| `file_003` | `02da9a637b4afc60` | `ddd1dfc6aa7c86cf` | `.pdf` | `accepted` | `written` | `spacy` | `0.83` | `usable_with_review` | `low_text_density, accepted_clean_input` | `` |
| `file_004` | `8bca6acbe5829fbe` | `472ef703930bee39` | `.pdf` | `review_ocr_quality` | `queued_for_review` | `rules_based` | `0.45` | `poor_ocr` | `poor_input_ocr, low_text_density, table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities` | `` |
| `file_005` | `651d7154030f7d51` | `acb6ac09b7f9e9de` | `.pdf` | `review` | `queued_for_review` | `rules_based` | `0.45` | `usable_with_review` | `low_text_density, table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities` | `` |
| `file_006` | `c354c20c9ad30f13` | `d387653b5a4e4d9b` | `.pdf` | `accepted` | `written` | `spacy` | `0.767` | `good` | `low_text_density, accepted_clean_input, lab_report_detected` | `` |
| `file_007` | `28fd4495f6687ae0` | `41ab88e8cce8d4b7` | `.pdf` | `accepted` | `written` | `spacy` | `0.9` | `good` | `low_text_density, accepted_clean_input, lab_report_detected` | `` |
| `file_008` | `8a25bbaf71f5c62f` | `c9884a65602107d7` | `.pdf` | `accepted` | `written` | `spacy` | `0.9` | `good` | `low_text_density, table_structure_loss, accepted_clean_input, lab_report_detected` | `` |
| `file_009` | `135075b739b67c18` | `abe1e8f2ffb9df86` | `.pdf` | `accepted` | `written` | `spacy` | `0.83` | `usable_with_review` | `low_text_density, accepted_clean_input` | `` |
| `file_010` | `7173c4713359ce65` | `3fcf796bf006eaa4` | `.pdf` | `accepted` | `written` | `spacy` | `0.9` | `good` | `low_text_density, accepted_clean_input, lab_report_detected` | `` |
| `file_011` | `df6b0747949611fb` | `944cafc4673bda98` | `.pdf` | `accepted` | `written` | `spacy` | `0.838` | `good` | `low_text_density, accepted_clean_input, lab_report_detected` | `` |
| `file_012` | `f6894f2ce0e360c4` | `625c56f3f902c0a2` | `.pdf` | `accepted` | `written` | `spacy` | `0.9` | `good` | `low_text_density, accepted_clean_input, lab_report_detected` | `` |
| `file_013` | `991607489f76fee7` | `344c35d44a9f7a2e` | `.pdf` | `review` | `queued_for_review` | `spacy` | `0.625` | `usable_with_review` | `low_text_density, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities` | `` |
| `file_014` | `b2d437eb93736785` | `537fc50e911a18d7` | `.pdf` | `accepted` | `written` | `spacy` | `0.83` | `usable_with_review` | `low_text_density, accepted_clean_input` | `` |
| `file_015` | `d2a2b96645ec5636` | `57cb83e69a6fdde5` | `.pdf` | `accepted` | `written` | `spacy` | `0.7` | `usable_with_review` | `low_text_density, accepted_clean_input` | `` |
