# MEDAI-DOC-TYPE-EVAL-01-FIX

## Conclusion

`medai_doc_type_eval_01_fix_ready`

## Root Cause / Observed Issue

The first local batch evaluation report showed high Unknown coverage and one Unknown plus accepted status inconsistency, but it did not make that inconsistency actionable. The harness also needed safe Unknown grouping so the next work can distinguish text-visibility issues, fallback misses, generic cues, ambiguity, and status mapping anomalies.

## What Changed

- Added status consistency checks.
- Unknown plus accepted is now flagged as a status anomaly.
- Accepted count is reported separately from auto-accept allowed count.
- Accepted status source is recorded as runtime validation status, runtime outcome, prior record status, or not accepted.
- Unknown records receive a safe failure bucket.
- Aggregate Unknown diagnostics are reported.
- A priority Unknown sample list is emitted with anonymous IDs only.
- Recommendation logic now calls out status mapping, OCR/text visibility, cue-pack review, external API violation, and auto-accept violation cases.

## Unknown Diagnostic Buckets

- `no_safe_document_family_cues`
- `insufficient_text_visibility`
- `OCR_not_triggered`
- `fallback_ran_but_no_family_match`
- `generic_cues_only`
- `ambiguous_below_threshold`
- `unsupported_or_empty_text`
- `status_mapping_anomaly`

## Safety Boundary

- Runtime behavior changed: false.
- OCR routing changed: false.
- OCR engine changed: false.
- Document-family classifier logic changed: false.
- Cue thresholds changed: false.
- Conflict rules changed: false.
- Auto-acceptance changed: false.
- Clinical interpretation added: false.
- Imaging interpretation added: false.
- Medication parsing added: false.
- Dose parsing added: false.
- Lab value parsing added: false.
- DDI logic changed: false.
- B07 changed: false.
- ROUTE-FIX changed: false.
- DB schema changed: false.
- Command allowlist changed: false.
- External API behavior changed: false.

## Validation Results

- `python -m pytest tests/test_medai_doc_type_eval_01_fix_status_unknown.py tests/test_medai_doc_type_eval_01_batch_harness.py -q` — passed, 12 passed, 1 warning.
- `python -m pytest tests/test_medai_doc_type_family_01_fix2_imaging_treatment_conflict.py tests/test_medai_doc_type_family_01_fix_runtime_imaging_propagation.py tests/test_medai_doc_type_family_01.py tests/test_medai_ui_run_review_polish_01_fix_state_consistency.py tests/test_medai_ui_run_review_polish_01.py -q` — passed, 34 passed.
- `python -m pytest tests/test_medai_ru_treatment_doc_type_01_fix2_always_emit_diagnostic.py tests/test_medai_ru_treatment_doc_type_01_fix_runtime_propagation.py tests/test_medai_ru_treatment_doc_type_01.py tests/test_medai_ru_lab_ocr_gate_02b_fix2_real_format_cue_coverage.py tests/test_medai_ru_lab_ocr_gate_02b_fix_fallback_cue_diagnostic.py tests/test_medai_ru_lab_ocr_gate_02b_local_cyrillic_fallback.py tests/test_medai_ru_lab_ocr_gate_02a_runtime_propagation.py tests/test_medai_ru_lab_ocr_gate_02a_shadow_marker.py tests/test_medai_ru_lab_ocr_gate_01.py tests/test_medai_ru_lab_text_vis_01.py tests/test_medai_ru_lab_extract_diag_01.py tests/test_medai_ru_doc_type_02_real_text_path_diagnostic.py tests/test_medai_ru_doc_type_01_russian_detection.py tests/test_medai_lab_fix_02_general_lab_detection.py tests/test_medai_lab_fix_01_document_type_review_reason.py -q` — passed, 128 passed.
- `python scripts/run_cka_final_mvp_release_validation.py` — passed, 12/12 cases, 693 tests passed.
- `python scripts/run_b07_term01_opt_in_integration_validation.py` — passed, 6/6 cases, external_api_used false.
- `python scripts/run_medai_route_fix01_validation.py` — passed.
- `python -m pytest tests` — passed, 2498 passed, 4 skipped, 22 warnings in 1685.48 seconds.

## Privacy

- External API used: false.
- Raw OCR text in public reports: false.
- Raw document text in public reports: false.
- Raw filenames or private paths in public reports: false.
- PHI in public reports: false.
- Secrets in public reports: false.
- Source documents staged: false.
- Runtime DB staged: false.

## Next Step

Rerun the local batch evaluation on the private folder and inspect Unknown diagnostic buckets plus status consistency before changing any cue packs.
