# MEDAI-AI-FIRST-CORPUS-TOKENIZATION-RERUN-COMMIT-AND-BLOCKER-TRIAGE-17A-R1

## Scope

This sanitized triage uses existing 17A public status artifacts only. It includes counts only and excludes raw filenames, private identifier values, raw OCR text, token maps, tokenized corpus payloads, and provider responses.

## Safety

- provider_call_made: `false`
- billing_api_call_made: `false`
- live_gate_set: `false`
- mkb_db_opened: `false`
- active_mkb_write: `false`
- auto_accept_enabled: `false`
- medical_decision_made: `false`
- private_values_included: `false`
- raw_ocr_included: `false`
- token_maps_included: `false`
- raw_filenames_included: `false`

## Corpus Counts

- vault_status: `ACTIVE`
- tokenized_corpus_generated: `True`
- total_files_seen: `599`
- files_tokenized: `12`
- ready_files: `12`
- blocked_files: `587`
- supported_files: `597`
- unsupported_files: `2`
- duplicate_files: `88`

## Blocked By Reason

- duplicate: `88`
- extraction_unavailable: `497`
- unsupported: `2`

## Required Blocker Classes

- unsupported_by_extension_or_type: `2`
- extraction_unavailable: `497`
- extraction_failed: `0`
- residual_pi_pattern_count_gt_0: `0`
- duplicate_files: `88`
- empty_or_near_empty_extracted_text: `0`
- ocr_not_available: `0`
- path_or_folder_issues: `0`

## Unsupported By Extension Or Type

- .docx: `3`
- .jpg: `10`
- .mp3: `2`
- .pdf: `576`
- .rtf: `3`
- .tif: `5`

## Interpretation

The active vault enabled tokenization and produced 12 ready files. The remaining blocked set is dominated by extraction availability, especially PDF rows marked `extraction_unavailable` with `pdf_text_extraction_unavailable`. The next block should run 17B dry-run on the ready subset only while treating extraction/OCR support as a separate blocker-reduction effort.
