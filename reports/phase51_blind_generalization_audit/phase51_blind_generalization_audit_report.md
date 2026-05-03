# Phase 51 Blind PDF Generalization Audit

- Timestamp: `2026-05-03T02:05:11.644535+00:00`
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
- Report PDF artifacts tracked: `False`
- Conclusion: `PASS_SAFETY_WEAK_AUTOMATION`

## Per-File Results

| File ID | Ext | Size | Status | Outcome | Extractor | Confidence | OCR band | Reason codes | Error |
| --- | --- | ---: | --- | --- | --- | ---: | --- | --- | --- |
| `file_0001_27076f66ce5c` | `.pdf` | 65522 | `accepted` | `written` | `spacy` | `0.9` | `good` | `low_text_density, accepted_clean_input` | `` |
| `file_0002_f31e2362f04d` | `.pdf` | 147314 | `accepted` | `written` | `spacy` | `0.838` | `good` | `low_text_density, accepted_clean_input, lab_report_detected` | `` |
| `file_0003_1a6d5de8d958` | `.pdf` | 52720 | `accepted` | `written` | `spacy` | `0.83` | `usable_with_review` | `low_text_density, accepted_clean_input` | `` |
| `file_0004_7dec27b556bb` | `.pdf` | 33372 | `review_ocr_quality` | `queued_for_review` | `rules_based` | `0.45` | `poor_ocr` | `poor_input_ocr, low_text_density, table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities` | `` |
| `file_0005_2b4bce9cc59b` | `.pdf` | 32373 | `review` | `queued_for_review` | `rules_based` | `0.45` | `usable_with_review` | `low_text_density, table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities` | `` |
| `file_0006_3546164e66e7` | `.pdf` | 61939 | `accepted` | `written` | `spacy` | `0.767` | `good` | `low_text_density, accepted_clean_input, lab_report_detected` | `` |
| `file_0007_3104e6d26574` | `.pdf` | 45908 | `accepted` | `written` | `spacy` | `0.9` | `good` | `low_text_density, accepted_clean_input, lab_report_detected` | `` |
| `file_0008_3b38c741e3e2` | `.pdf` | 91293 | `accepted` | `written` | `spacy` | `0.9` | `good` | `low_text_density, table_structure_loss, accepted_clean_input, lab_report_detected` | `` |
| `file_0009_5a32461167da` | `.pdf` | 16269 | `accepted` | `written` | `spacy` | `0.83` | `usable_with_review` | `low_text_density, accepted_clean_input` | `` |
| `file_0010_52632041ce20` | `.pdf` | 54899 | `accepted` | `written` | `spacy` | `0.9` | `good` | `low_text_density, accepted_clean_input, lab_report_detected` | `` |
| `file_0011_465308a89705` | `.pdf` | 71391 | `accepted` | `written` | `spacy` | `0.838` | `good` | `low_text_density, accepted_clean_input, lab_report_detected` | `` |
| `file_0012_20f04b7d3760` | `.pdf` | 30912 | `accepted` | `written` | `spacy` | `0.9` | `good` | `low_text_density, accepted_clean_input, lab_report_detected` | `` |
| `file_0013_677547654424` | `.pdf` | 14449 | `review` | `queued_for_review` | `spacy` | `0.625` | `usable_with_review` | `low_text_density, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities` | `` |
| `file_0014_69c1c4959232` | `.pdf` | 39397 | `accepted` | `written` | `spacy` | `0.83` | `usable_with_review` | `low_text_density, accepted_clean_input` | `` |
| `file_0015_c430dfa06ebb` | `.pdf` | 14803 | `accepted` | `written` | `spacy` | `0.7` | `usable_with_review` | `low_text_density, accepted_clean_input` | `` |
