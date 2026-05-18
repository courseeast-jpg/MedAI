# MEDAI-DOC-TYPE-EVAL-04

## Conclusion

`medai_doc_type_eval_04_language_script_detector_ready`

Scope: evaluation/reporting and language/script visibility diagnostics only.

## Root Cause Analysis

The post-EVAL-03 80-file batch showed that the next bottleneck is `detector_returned_unknown`.

Safe aggregate evidence:

- Total files evaluated: `80`
- Lab result: `15`
- Urinalysis: `1`
- Unknown: `64`
- External API used: `0`
- Accepted after normalization: `0`
- Auto-accept allowed: `0`
- Unknown `insufficient_text_visibility`: `62`
- Unknown `fallback_ran_but_no_family_match`: `1`
- Unknown `status_mapping_anomaly`: `1`
- Not fallback eligible because `language_visibility_unknown`: `63`
- Language visibility unknown reason `detector_returned_unknown`: `63`
- Language visibility unknown reason `no_text_available`: `1`

This points to language/script detector robustness and reporting visibility, not OCR routing, classifier cue-pack coverage, extraction errors, or image-like PDFs bypassing OCR.

## What Changed

The batch evaluation harness now records safe detector buckets for language/script unknowns:

- `detector_input_empty`
- `detector_input_tiny`
- `detector_input_numeric_heavy`
- `detector_input_symbol_heavy`
- `detector_input_mixed_script`
- `detector_input_garbled_or_mojibake`
- `detector_confidence_below_threshold`
- `detector_output_not_propagated`
- `script_detectable_language_unknown`
- `detector_unknown_unclassified`

It also adds aggregate counts for:

- `numeric_heavy`
- `alphabetic_low`
- `cyrillic_present_but_language_unknown`
- `latin_present_but_language_unknown`
- `mixed_script`
- `detector_output_missing`
- `detector_low_confidence`

## Script-Level Visibility

When language remains unknown but script is clearly visible, the report now uses a conservative script-level visibility label:

- `cyrillic_visible_language_unknown`
- `latin_visible_language_unknown`
- `mixed_script_visible_language_unknown`

This is reporting-only. It does not classify document family from script alone, does not make files OCR-eligible, and does not alter OCR routing.

## Report Additions

The batch report now includes:

- `language_script_detector_unknown_diagnostics`
- `language_script_detector_unknown_bucket`
- `language_script_visibility`
- `raw_language_visibility_status`
- `alphabetic_content_bucket`
- `numeric_content_bucket`
- `symbol_content_bucket`
- `garbled_text_detected`
- `detector_confidence_bucket`
- `top_detector_unknown_samples`

All examples remain anonymous safe IDs only.

## Recommendation Logic

- Numeric-heavy/table text: OCR gate review may be needed later.
- Cyrillic visible but language unknown: inspect language detector confidence or metadata propagation.
- Low alphabetic content: keep Unknown/manual review or evaluate OCR gate expansion in a later block.
- Detector output missing: inspect detector output propagation.
- Garbled text: investigate extraction quality before cue-pack changes.

## Validation Results

- `python -m pytest tests/test_medai_doc_type_eval_04_language_script_detector.py tests/test_medai_doc_type_eval_03_status_language_visibility.py tests/test_medai_doc_type_eval_02_unknown_text_visibility.py tests/test_medai_doc_type_eval_01_fix_status_unknown.py tests/test_medai_doc_type_eval_01_batch_harness.py -q` -> `30 passed, 1 warning`
- `python -m pytest tests/test_medai_doc_type_family_01_fix2_imaging_treatment_conflict.py tests/test_medai_doc_type_family_01_fix_runtime_imaging_propagation.py tests/test_medai_doc_type_family_01.py tests/test_medai_ui_run_review_polish_01_fix_state_consistency.py tests/test_medai_ui_run_review_polish_01.py -q` -> `34 passed`
- OCR gate/text visibility/Russian document-type regression group -> `128 passed`
- `python scripts/run_cka_final_mvp_release_validation.py` -> `PASS; 12/12 cases passed; 693 tests passed; external API used false`
- `python scripts/run_b07_term01_opt_in_integration_validation.py` -> `PASS; 6/6 cases passed; external_api_used false`
- `python scripts/run_medai_route_fix01_validation.py` -> `PASS; medai_route_fix01_ready; external_api_used false`
- `python -m pytest tests` -> `2516 passed, 4 skipped, 22 warnings in 1687.46s`

## Safety / Privacy

- OCR routing changed: `false`
- OCR engine changed: `false`
- Classifier changed: `false`
- Cue pack changed: `false`
- Document-family thresholds changed: `false`
- Conflict rules changed: `false`
- Auto-acceptance logic changed: `false`
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
- Private files staged: `false`
- Runtime DB staged: `false`
