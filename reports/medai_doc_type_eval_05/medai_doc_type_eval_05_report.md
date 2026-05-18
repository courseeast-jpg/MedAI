# MEDAI-DOC-TYPE-EVAL-05

## Conclusion

`medai_doc_type_eval_05_script_visible_unknown_cue_shapes_ready`

Scope: evaluation/reporting only.

## Root Cause Hypothesis

The post-EVAL-04 80-file batch showed that the current bottleneck is script-visible Unknown records with no matched document-family cue keys, not OCR routing.

Safe aggregate evidence:

- Total files evaluated: `80`
- Lab result: `15`
- Urinalysis: `1`
- Unknown: `64`
- Accepted after normalization: `0`
- Auto-accept allowed: `0`
- External API used: `0`
- Detector returned unknown: `63`
- Detector unknown bucket `script_detectable_language_unknown`: `63`
- Script-level visibility `latin_visible_language_unknown`: `80`
- Image-like PDFs not routed to OCR: `0`
- Text-layer PDFs with too little text: `0`
- Extraction errors: `0`
- Fallback eligible but not triggered: `0`

EVAL-05 therefore adds a safe text-shape and cue-audit layer for script-visible Unknown files so the next block can distinguish likely family-structure gaps from generic forms, header noise, or true Unknown/manual-review cases.

## Cue-Audit Buckets Added

Per-file records now include only safe category buckets:

- `dominant_script`
- `alphabetic_content_bucket`
- `numeric_content_bucket`
- `table_like_structure_detected`
- `section_heading_shape_detected`
- `medical_abbreviation_shape_detected`
- `date_or_schedule_shape_detected`
- `imaging_modality_shape_detected`
- `lab_table_shape_detected`
- `administrative_form_shape_detected`
- `cue_audit_result`

Supported `cue_audit_result` values:

- `no_known_family_shapes`
- `generic_form_shapes_only`
- `possible_lab_shape_without_language_cues`
- `possible_imaging_shape_without_language_cues`
- `possible_treatment_shape_without_language_cues`
- `likely_nonmedical_or_header_noise`
- `needs_manual_review`

## Aggregate Counts Added

The batch report now summarizes:

- Script-visible Unknown with table-like shape
- Script-visible Unknown with imaging-like shape
- Script-visible Unknown with lab-like shape
- Script-visible Unknown with treatment/schedule-like shape
- Script-visible Unknown with only generic/admin shapes
- Script-visible Unknown with no known shapes
- Cue-audit result bucket counts

## Recommendation Logic

- Many `possible_lab_shape_without_language_cues`: recommend conservative Latin/structure lab cue audit in a later block.
- Many `possible_imaging_shape_without_language_cues`: recommend imaging cue audit in a later block.
- Many `possible_treatment_shape_without_language_cues`: recommend treatment/schedule cue audit in a later block.
- Many `no_known_family_shapes`: leave Unknown/manual review.
- Many `likely_nonmedical_or_header_noise`: review native-text sufficiency before any OCR-routing change.
- Useful shape buckets with no family labels: recommend cue-family update in a later block, not in this diagnostic block.

Status anomaly handling from EVAL-03 is preserved: Unknown runtime accepted status remains normalized to review and `invalid_status_mapping_normalized` remains reported.

## Validation Results

- `python -m pytest tests/test_medai_doc_type_eval_05_script_visible_unknown_cue_shapes.py -q` -> `6 passed, 1 warning`
- `python -m pytest tests/test_medai_doc_type_eval_05_script_visible_unknown_cue_shapes.py tests/test_medai_doc_type_eval_04_language_script_detector.py tests/test_medai_doc_type_eval_03_status_language_visibility.py tests/test_medai_doc_type_eval_02_unknown_text_visibility.py tests/test_medai_doc_type_eval_01_fix_status_unknown.py tests/test_medai_doc_type_eval_01_batch_harness.py -q` -> `36 passed, 1 warning`
- `python -m pytest tests/test_medai_doc_type_family_01_fix2_imaging_treatment_conflict.py tests/test_medai_doc_type_family_01_fix_runtime_imaging_propagation.py tests/test_medai_doc_type_family_01.py tests/test_medai_ui_run_review_polish_01_fix_state_consistency.py tests/test_medai_ui_run_review_polish_01.py -q` -> `34 passed`
- OCR gate/text visibility/Russian document-type regression group -> `128 passed`
- `python scripts/run_cka_final_mvp_release_validation.py` -> `PASS; 12/12 cases passed; 693 tests passed; external API used false`
- `python scripts/run_b07_term01_opt_in_integration_validation.py` -> `PASS; 6/6 cases passed; external_api_used false`
- `python scripts/run_medai_route_fix01_validation.py` -> `PASS; medai_route_fix01_ready; external_api_used false`
- `python -m pytest tests` -> `2522 passed, 4 skipped, 22 warnings in 1735.35s`

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
