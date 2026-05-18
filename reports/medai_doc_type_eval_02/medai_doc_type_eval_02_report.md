# MEDAI-DOC-TYPE-EVAL-02

## Conclusion

`medai_doc_type_eval_02_unknown_text_visibility_ocr_routing_diagnostic_ready`

Scope: evaluation/reporting only.

## Observed Issue

The first 80-file local batch evaluation showed:

- Total files evaluated: `80`
- Unknown: `64`
- Unknown with low or incomplete text visibility: `64`
- Unknown with no family cue keys: `64`
- Unknown with fallback false: `63`
- Fallback ran but no family match: `1`
- Unknown accepted anomaly: `1`
- External API used: `0`
- Auto-accept allowed: `0`

The previous report was useful, but it did not explain whether fallback=false Unknown files were caused by missing text layers, short native text, image-like PDFs not routed to OCR, fallback ineligibility, extraction errors, or language visibility gaps.

## What Changed

The batch evaluation harness now records safe OCR-routing diagnostics for Unknown files:

- `text_extraction_status_bucket`
- `native_text_length_bucket`
- `page_count_bucket`
- `pdf_text_layer_detected`
- `image_like_pdf`
- `ocr_fallback_eligible`
- `ocr_fallback_not_triggered_reason`
- `language_visibility_status`
- `cyrillic_visibility_status`
- `document_family_cue_count_bucket`
- `unknown_ocr_routing_bucket`

Fallback=false Unknown files are grouped into safe buckets:

- `no_text_layer`
- `text_layer_present_but_too_short`
- `image_like_pdf_but_not_routed_to_ocr`
- `language_visibility_unknown`
- `unsupported_pdf_structure`
- `extraction_error`
- `routing_not_eligible`
- `unknown_reason`

## Aggregate Diagnostics Added

The report now includes:

- Unknown image-like PDFs not routed to OCR
- Unknown text-layer PDFs with too little text
- Unknown files with extraction errors
- Unknown files eligible for fallback but fallback not triggered
- Unknown files not eligible for fallback and why
- Top anonymous Unknown samples by diagnostic priority

## Status Anomaly Handling

Unknown + accepted is now explicitly flagged as a status anomaly. The report separates:

- accepted count
- auto-accept allowed count
- accepted status source

For the observed batch, the anomaly should be carried forward as `accepted_source=runtime_validation_status` and investigated in a status-mapping follow-up. This block does not fix status mapping.

## Validation Results

- `python -m pytest tests/test_medai_doc_type_eval_02_unknown_text_visibility.py tests/test_medai_doc_type_eval_01_fix_status_unknown.py tests/test_medai_doc_type_eval_01_batch_harness.py -q` -> `17 passed, 1 warning`
- `python -m pytest tests/test_medai_doc_type_family_01_fix2_imaging_treatment_conflict.py tests/test_medai_doc_type_family_01_fix_runtime_imaging_propagation.py tests/test_medai_doc_type_family_01.py tests/test_medai_ui_run_review_polish_01_fix_state_consistency.py tests/test_medai_ui_run_review_polish_01.py -q` -> `34 passed`
- OCR gate/text visibility/Russian document-type regression group -> `128 passed`
- `python scripts/run_cka_final_mvp_release_validation.py` -> `PASS; 12/12 cases passed; 693 tests passed; external API used false`
- `python scripts/run_b07_term01_opt_in_integration_validation.py` -> `PASS; 6/6 cases passed; external_api_used false`
- `python scripts/run_medai_route_fix01_validation.py` -> `PASS; medai_route_fix01_ready; external_api_used false`
- `python -m pytest tests` -> `2503 passed, 4 skipped, 22 warnings in 1671.15s`

## Safety / Privacy

- Runtime behavior changed: `false`
- OCR routing changed: `false`
- OCR engine changed: `false`
- Document-family classifier logic changed: `false`
- Cue thresholds changed: `false`
- Conflict rules changed: `false`
- Auto-acceptance changed: `false`
- Clinical interpretation added: `false`
- Imaging interpretation added: `false`
- Medication parsing added: `false`
- Dose parsing added: `false`
- Lab value parsing added: `false`
- DDI logic changed: `false`
- B07 changed: `false`
- ROUTE-FIX changed: `false`
- DB schema changed: `false`
- Command allowlist changed: `false`
- External API enabled: `false`
- Raw OCR text in public reports: `false`
- Raw document text in public reports: `false`
- Raw filenames/private paths in public reports: `false`
- Source documents staged: `false`
