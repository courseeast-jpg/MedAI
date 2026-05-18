# MEDAI-DOC-TYPE-EVAL-03

## Conclusion

`medai_doc_type_eval_03_status_mapping_language_visibility_ready`

Scope: evaluation/reporting plus batch-eval status mapping only.

## Observed Issue

The 80-file batch showed:

- Total files evaluated: `80`
- Unknown: `64`
- Unknown with low or incomplete text visibility: `64`
- Unknown with no family cue keys: `64`
- Unknown with fallback false: `63`
- Not fallback eligible reason `language_visibility_unknown`: `63`
- OCR fallback eligible but not triggered: `0`
- Image-like PDFs not routed to OCR: `0`
- Extraction errors: `0`
- Unknown accepted anomaly count: `1`

The anomalous safe record was `file_001`: document type `Unknown`, report status `accepted`, accepted source `runtime_validation_status`.

## Status Mapping Fix

Root cause: the batch evaluation harness derived `review_status` directly from runtime `validation_status` before applying document-type review-bound semantics. That allowed `Unknown + runtime_validation_status=accepted` to appear as accepted in the report.

Fix:

- Unknown runtime-accepted records are normalized to `review`.
- The original raw status is preserved as `raw_review_status`.
- The source remains visible as `accepted_status_source`.
- The record is flagged with `invalid_status_mapping_normalized`.
- Historical/prior accepted status can still be explained separately as prior status.
- `accepted_count` remains separate from `auto_accept_allowed_count`.

Known accepted non-Unknown document types are not changed by this mapping.

## Language Visibility Audit

Added safe per-file fields:

- `text_source_present`
- `text_extraction_attempted`
- `text_extraction_result_bucket`
- `language_detector_attempted`
- `language_detector_input_bucket`
- `script_detection_attempted`
- `script_detection_result`
- `visibility_unknown_reason`

Supported visibility unknown reasons:

- `no_text_available`
- `text_not_passed_to_visibility_detector`
- `detector_not_called`
- `detector_returned_unknown`
- `numeric_or_symbol_only_text`
- `metadata_missing`
- `unknown`

Added aggregate counts for each visibility unknown reason so the next block can decide whether to investigate text propagation, language detection metadata, or a later OCR-routing change.

## Validation Results

- `python -m pytest tests/test_medai_doc_type_eval_03_status_language_visibility.py tests/test_medai_doc_type_eval_02_unknown_text_visibility.py tests/test_medai_doc_type_eval_01_fix_status_unknown.py tests/test_medai_doc_type_eval_01_batch_harness.py -q` -> `23 passed, 1 warning`
- `python -m pytest tests/test_medai_doc_type_family_01_fix2_imaging_treatment_conflict.py tests/test_medai_doc_type_family_01_fix_runtime_imaging_propagation.py tests/test_medai_doc_type_family_01.py tests/test_medai_ui_run_review_polish_01_fix_state_consistency.py tests/test_medai_ui_run_review_polish_01.py -q` -> `34 passed`
- OCR gate/text visibility/Russian document-type regression group -> `128 passed`
- `python scripts/run_cka_final_mvp_release_validation.py` -> `PASS; 12/12 cases passed; 693 tests passed; external API used false`
- `python scripts/run_b07_term01_opt_in_integration_validation.py` -> `PASS; 6/6 cases passed; external_api_used false`
- `python scripts/run_medai_route_fix01_validation.py` -> `PASS; medai_route_fix01_ready; external_api_used false`
- `python -m pytest tests` -> `2509 passed, 4 skipped, 22 warnings in 1711.60s`

## Safety / Privacy

- OCR routing changed: `false`
- OCR engine changed: `false`
- Classifier changed: `false`
- Cue pack changed: `false`
- Cue thresholds changed: `false`
- Conflict rules changed: `false`
- Auto-accept expanded: `false`
- Clinical interpretation added: `false`
- Imaging interpretation added: `false`
- Medication parsing added: `false`
- Dose/frequency/duration parsing added: `false`
- Lab value parsing added: `false`
- DDI logic changed: `false`
- B07 changed: `false`
- ROUTE-FIX changed: `false`
- DB schema changed: `false`
- Command allowlist changed: `false`
- External API changed: `false`
- Raw OCR text in public reports: `false`
- Raw document text in public reports: `false`
- Raw filenames/private paths in public reports: `false`
- Source documents staged: `false`
