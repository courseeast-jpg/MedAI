# Phase64 RTF Local Text Parser Prototype

- Generated at: `2026-05-04T22:03:07.944696+00:00`
- RTF supported extension registered: `True`
- RTF file count: `3`
- RTF counts: `{"accepted": 0, "empty": 0, "errors": 0, "processing_error": 0, "review": 3, "review_ocr_quality": 0, "supported_processed": 3, "total": 3, "unsupported_extension": 0}`
- Non-RTF extensions left unsupported/excluded: `{".docx": 3, ".mp3": 3, ".msg": 1, ".ogg": 1}`
- External API used: `False`
- Local-only forced: `True`
- Raw PHI logged in public reports: `False`
- Reconciliation passed: `True`
- Production safety regression: `False`
- Conclusion: `rtf_local_parser_prototype_ready`

## RTF Files

| Safe File ID | Filename Hash | Content Hash | File Type | Status | Extractor | Confidence | Reason Codes |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| `corpus_file_000153` | `4f53f7f17f5d2c9f` | `31691fdce27ff72e` | `rtf_text` | `review` | `phi3` | `0.422` | `extraction_low_confidence, safety_gate_low_confidence` |
| `corpus_file_000155` | `4a8c737641146134` | `520c7c496a20b8ce` | `rtf_text` | `review` | `rules_based` | `0.45` | `table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities` |
| `corpus_file_000156` | `9458e54263b281e8` | `11fa1fffc429bdba` | `rtf_text` | `review` | `rules_based` | `0.45` | `table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities` |

## Safety

- Public reports use safe file IDs and hashes only.
- RTF text is parsed locally and is not written to public reports.
- `.docx`, `.msg`, `.mp3`, and `.ogg` are not enabled in this phase.
- Existing PDF/TXT/image behavior and confidence gates are unchanged.
