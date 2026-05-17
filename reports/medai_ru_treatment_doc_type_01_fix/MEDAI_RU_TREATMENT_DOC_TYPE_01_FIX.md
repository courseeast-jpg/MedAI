# MEDAI-RU-TREATMENT-DOC-TYPE-01-FIX

Conclusion: medai_ru_treatment_doc_type_01_fix_ready

This block fixes runtime propagation for fallback-recovered Russian treatment and medication schedule classification metadata. It adds a separate treatment classification diagnostic to the local OCR fallback record and carries that safe diagnostic into Run & Review results.

## Runtime Propagation Result

- The local fallback OCR path now emits `ocr_gate_fallback_treatment_classification_diagnostic`.
- Run & Review result assembly now preserves that diagnostic in the per-file record.
- If fallback diagnostics identify `Medication plan` or `Treatment plan`, Run & Review can use that candidate when the primary runtime document type is missing or unknown.

## Safe Before / After Metadata

| Safe case | Before | After |
| --- | --- | --- |
| schedule_document | `matched_lab_cue_keys: []`, `matched_document_type_candidate: Unknown` | treatment diagnostic includes safe schedule cue keys and candidate `Medication plan` or `Treatment plan` |
| lab_document | safe lab cue keys classify as `Lab result` | remains `Lab result` |
| non_medical_cyrillic | `Unknown` | remains `Unknown` |

## Treatment Cue Keys Propagated

- medication_schedule_header
- date_grid
- physiotherapy_section
- diet_recommendation_section
- administration_schedule_pattern

## Safety

- external_api_used: false
- raw_ocr_text_in_public_reports: false
- medication_interpretation_added: false
- dose_parsing_added: false
- ddi_logic_changed: false
- auto_accept_expanded: false
- affected_files_remain_review_bound: true
- no raw PHI
- no raw filenames
- no raw OCR text
- no raw document text
- no private absolute paths
- no secrets

## Validation Results

- `python -m pytest tests/test_medai_ru_treatment_doc_type_01_fix_runtime_propagation.py -q`: 7 passed
- `python -m pytest tests/test_medai_ru_treatment_doc_type_01.py -q`: 7 passed
- `python -m pytest tests/test_medai_ru_lab_ocr_gate_02b_fix2_real_format_cue_coverage.py tests/test_medai_ru_lab_ocr_gate_02b_fix_fallback_cue_diagnostic.py tests/test_medai_ru_lab_ocr_gate_02b_local_cyrillic_fallback.py -q`: 22 passed
- `python -m pytest tests/test_medai_ru_lab_ocr_gate_02a_runtime_propagation.py tests/test_medai_ru_lab_ocr_gate_02a_shadow_marker.py tests/test_medai_ru_lab_ocr_gate_01.py tests/test_medai_ru_lab_text_vis_01.py tests/test_medai_ru_lab_extract_diag_01.py -q`: 43 passed
- `python -m pytest tests/test_medai_ru_doc_type_02_real_text_path_diagnostic.py tests/test_medai_ru_doc_type_01_russian_detection.py tests/test_medai_lab_fix_02_general_lab_detection.py tests/test_medai_lab_fix_01_document_type_review_reason.py tests/test_medai_ui_bugfix_02_readd_after_queue_clear.py tests/test_medai_ui_bugfix_01_clear_report_upload_idempotency.py -q`: 54 passed
- `python scripts/run_medai_ui_ops_panel_validation.py`: passed
- `python scripts/run_medai_ui_boot_fix_validation.py`: passed
- `python scripts/run_cka_final_mvp_release_validation.py`: passed, 12/12, 693 tests
- `python scripts/run_b07_term01_opt_in_integration_validation.py`: passed, 6/6
- `python scripts/run_medai_route_fix01_validation.py`: passed
- `python -m pytest tests`: 2445 passed, 4 skipped, 22 warnings
