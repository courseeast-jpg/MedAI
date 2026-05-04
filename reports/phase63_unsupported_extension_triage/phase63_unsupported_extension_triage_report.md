# Phase63 Unsupported Extension Triage

- Generated: `2026-05-04T21:15:56.071387+00:00`
- Unsupported count: `11`
- Local-only forced: `True`
- External API used: `False`
- Raw PHI logged in public reports: `False`
- Production extractor should change yet: `False`
- Conclusion: `triage_complete_support_prototype_recommended`

## Extension Distribution

- `.docx`: `3`
- `.mp3`: `3`
- `.msg`: `1`
- `.ogg`: `1`
- `.rtf`: `3`

## Recommended Action By Extension

### `.docx`
- File count: `3`
- Classification: `support_later`
- Recommended action: `future_privacy_safe_document_conversion_phase`
- Safe file IDs: `corpus_file_000118, corpus_file_000140, corpus_file_000162`
- Guidance: Convert manually to PDF/TXT for now, or queue a future local-only office-document conversion phase.
- Rationale: Office documents are containers and may contain metadata/embedded media. Add support only in a separate privacy-reviewed phase.

### `.mp3`
- File count: `3`
- Classification: `explicit_exclusion`
- Recommended action: `exclude_audio_video_from_medai_document_ingest`
- Safe file IDs: `corpus_file_000154, corpus_file_000571, corpus_file_000606`
- Guidance: Audio/video transcription is out of scope for the current document OCR/Layout release.
- Rationale: Audio/video requires a separate local transcription pipeline and different privacy controls.

### `.msg`
- File count: `1`
- Classification: `explicit_exclusion`
- Recommended action: `exclude_container_format_from_medai_ingest`
- Safe file IDs: `corpus_file_000159`
- Guidance: Do not process this container directly. Export the relevant medical document to PDF/TXT/image first.
- Rationale: Container/email/archive formats can hide filenames, attachments, metadata, and nested PHI.

### `.ogg`
- File count: `1`
- Classification: `explicit_exclusion`
- Recommended action: `exclude_audio_video_from_medai_document_ingest`
- Safe file IDs: `corpus_file_000163`
- Guidance: Audio/video transcription is out of scope for the current document OCR/Layout release.
- Rationale: Audio/video requires a separate local transcription pipeline and different privacy controls.

### `.rtf`
- File count: `3`
- Classification: `safe_to_support_now`
- Recommended action: `support_prototype`
- Safe file IDs: `corpus_file_000153, corpus_file_000155, corpus_file_000156`
- Guidance: RTF/text-like format can be evaluated in a narrow local-only prototype. Keep output review-gated until validated.
- Rationale: Text-like formats can be parsed locally without cloud calls or OCR routing changes.

## Phase64 Recommendation

- Next phase: `Phase64 Narrow RTF/Text-Like Local Parser Prototype`
- Target extensions: `.rtf`
- Production extractor should change yet: `False`
- Reason: Only text-like unsupported extensions should be considered next, and only as a review-gated local prototype.

## Safe Unsupported Files

| Safe File ID | Filename Hash | Content Hash | Extension | Accounting Category |
| --- | --- | --- | --- | --- |
| `corpus_file_000118` | `51dd528d1c9ed01a` | `cf082bb5427571dd` | `.docx` | `unsupported_extension` |
| `corpus_file_000140` | `2d09c8e08333f62c` | `60156823f3bb0b96` | `.docx` | `unsupported_extension` |
| `corpus_file_000153` | `4f53f7f17f5d2c9f` | `31691fdce27ff72e` | `.rtf` | `unsupported_extension` |
| `corpus_file_000154` | `5ecffe05c774c200` | `4baee25c5093bbc6` | `.mp3` | `unsupported_extension` |
| `corpus_file_000155` | `4a8c737641146134` | `520c7c496a20b8ce` | `.rtf` | `unsupported_extension` |
| `corpus_file_000156` | `9458e54263b281e8` | `11fa1fffc429bdba` | `.rtf` | `unsupported_extension` |
| `corpus_file_000159` | `085e5586e54b1b3c` | `c43532b73b5f501f` | `.msg` | `unsupported_extension` |
| `corpus_file_000162` | `bf1d53c7844fe77d` | `e8d31a40ab5ab73b` | `.docx` | `unsupported_extension` |
| `corpus_file_000163` | `f58797a08a2e0a65` | `e551909c8abf77a4` | `.ogg` | `unsupported_extension` |
| `corpus_file_000571` | `32003e3ce34f9cf4` | `06af6e47559b51a7` | `.mp3` | `unsupported_extension` |
| `corpus_file_000606` | `3c5be31c540d831e` | `15687cac1ec8a8ad` | `.mp3` | `unsupported_extension` |
