# Phase53 Blind PDF Generalization Audit

- Timestamp: `2026-05-03T16:26:46.100444+00:00`
- Total files: `15`
- Processed files: `15`
- Accepted: `0`
- Review: `15`
- Review OCR quality: `1`
- Empty: `11`
- Errors: `0`
- Local-only mode: `True`
- External API default allowed: `False`
- Privacy gate status: `local_only`
- Raw PHI logged: `False`
- Raw PHI logged in public reports: `False`
- Report PDF artifacts tracked: `False`
- Conclusion: `PASS_SAFETY_WEAK_AUTOMATION`

## Per-File Results

| Safe File ID | Filename Hash | Content Hash | Type | Status | Outcome | Extractor | Confidence | OCR status | Reason codes | Error |
| --- | --- | ---: | --- | --- | --- | ---: | --- | --- | --- |
| `file_001` | `973978bafc8d72ff` | `118e1b8575ae47ab` | `pdf` | `review_ocr_quality` | `queued_for_review` | `spacy` | `0.425` | `good` | `table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities, classifier_legacy_ocr_flag, legacy_normalized_low_coverage` | `` |
| `file_002` | `02730b17c5d41a69` | `d6c779e738979ac5` | `pdf` | `review` | `queued_for_review` | `rules_based` | `0.45` | `good` | `table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities` | `` |
| `file_003` | `7ce31d001adf6272` | `ef6bbbf74f21a3bf` | `pdf` | `review` | `queued_for_review` | `rules_based` | `0.45` | `good` | `low_text_density, table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities` | `` |
| `file_004` | `1ff673421c6e0b5a` | `11b5ccb38fecfe81` | `pdf` | `review` | `queued_for_review` | `rules_based` | `0.45` | `good` | `table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities` | `` |
| `file_005` | `58fc59e09efaccd8` | `493d1d5be50439ee` | `pdf` | `review` | `queued_for_review` | `rules_based` | `0.45` | `good` | `low_text_density, table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities, lab_report_detected` | `` |
| `file_006` | `927ae81cbefc0ab9` | `59bc6b340d7a4d65` | `pdf` | `review` | `queued_for_review` | `rules_based` | `0.45` | `good` | `table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities, lab_report_detected` | `` |
| `file_007` | `73201a9ae16e54be` | `076ade41661d9056` | `pdf` | `review` | `queued_for_review` | `rules_based` | `0.45` | `good` | `table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities` | `` |
| `file_008` | `189b62913328d3c1` | `4ede2df7bce2aa4f` | `pdf` | `review` | `queued_for_review` | `rules_based` | `0.45` | `good` | `low_text_density, table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities` | `` |
| `file_009` | `91829b1417485ba9` | `5567b51fbf1a7d11` | `pdf` | `review` | `queued_for_review` | `rules_based` | `0.45` | `good` | `low_text_density, table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities, lab_report_detected` | `` |
| `file_010` | `8bf0b6ad44fd1ba4` | `50eb7f4db2a290b2` | `pdf` | `review` | `queued_for_review` | `rules_based` | `0.45` | `good` | `table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities, lab_report_detected` | `` |
| `file_011` | `f4db6f31d187f25d` | `0f6bf333b6d8b774` | `pdf` | `review` | `queued_for_review` | `rules_based` | `0.45` | `usable_with_review` | `table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities` | `` |
| `file_012` | `5d21385fbec9a500` | `ad7279062eca4200` | `pdf` | `review` | `queued_for_review` | `rules_based` | `0.45` | `good` | `table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities` | `` |
| `file_013` | `455c718547867264` | `1e79acda28bb30a9` | `image` | `review` | `queued_for_review` | `rules_based` | `0.45` | `good` | `low_text_density, table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities` | `` |
| `file_014` | `c4a491bc06a43cda` | `a9daf124b06e6a1c` | `pdf` | `review` | `queued_for_review` | `phi3` | `0.57` | `good` | `extraction_low_confidence, safety_gate_low_confidence, lab_report_detected` | `` |
| `file_015` | `841906333ffe78cc` | `32b86a2545270907` | `pdf` | `review` | `queued_for_review` | `phi3` | `0.0` | `good` | `low_text_density, table_structure_loss, extraction_low_confidence, safety_gate_low_confidence` | `` |
