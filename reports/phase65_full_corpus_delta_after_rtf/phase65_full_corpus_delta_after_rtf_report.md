# Phase65 Full Corpus Delta After RTF Support

- Generated at: `2026-05-04T22:47:20.975819+00:00`
- Before unsupported count: `11`
- After unsupported count: `8`
- Unsupported delta: `-3`
- RTF moved to supported processed: `True`
- RTF file count: `3`
- Non-RTF unsupported remaining: `{".docx": 3, ".mp3": 3, ".msg": 1, ".ogg": 1}`
- Accepted delta: `0`
- Review delta: `3`
- Empty delta: `0`
- Error delta: `-3`
- External API used: `False`
- Raw PHI logged: `False`
- Safety regression: `False`
- Reconciliation passed: `True`
- production_extractor_should_change_yet: `False`
- Conclusion: `rtf_delta_measured_no_safety_regression`

## RTF Records

| Safe File ID | Filename Hash | Content Hash | Status | Extractor | Confidence | Reason Codes |
| --- | --- | --- | --- | --- | ---: | --- |
| `corpus_file_000153` | `4f53f7f17f5d2c9f` | `31691fdce27ff72e` | `review` | `phi3` | `0.422` | `extraction_low_confidence, safety_gate_low_confidence` |
| `corpus_file_000155` | `4a8c737641146134` | `520c7c496a20b8ce` | `review` | `rules_based` | `0.45` | `table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities` |
| `corpus_file_000156` | `9458e54263b281e8` | `11fa1fffc429bdba` | `review` | `rules_based` | `0.45` | `table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities` |

## Recommendation

- Summary: RTF impact measured. Do not add more formats without prioritization evidence.
- DOCX: Support later only if operator review shows DOCX is important.
- MSG/audio: Keep MSG/audio explicitly excluded until a separate privacy-safe conversion/transcription phase is justified.
- Next class options: `pdf_ocr_low_quality, image_ocr_low_quality`
- Recommended next class: `pdf_ocr_low_quality`

## Safety

- No production extraction logic changed in Phase65.
- Public report uses safe IDs and hashes only.
- No filenames, paths, extracted text, OCR text, RTF text, or PHI are included.
