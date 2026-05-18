# MEDAI-DOC-TYPE-FAMILY-04 ? Larger Anonymized Batch Slice Validation

Conclusion: `medai_doc_type_family_04_larger_slice_validation_completed`.

## Input Slice

Temporary anonymized local slice of 507 supported files excluding the prior 80-file bang-folder subset; source names and paths were not recorded in public reports.

## Aggregate Counts

- Total files evaluated: `507`
- Lab result: `330`
- Imaging report: `14`
- Treatment plan: `4`
- Medication plan: `11`
- Clinical note: `2`
- Discharge summary: `0`
- Unknown: `107`
- Referral / Order: `0`
- Procedure report: `0`
- Pathology report: `0`
- Administrative / Insurance: `4`
- Urinalysis: `35`
- Review status counts: `{'review': 507}`
- Accepted count: `0`
- Auto-accept allowed count: `0`
- External API used count: `0`

## Unknown Summary

- Unknown count: `107`
- ambiguous_below_threshold: `15`
- fallback_ran_but_no_family_match: `17`
- insufficient_text_visibility: `75`

## False-Positive Risk Audit

- lab_vs_treatment_schedule_risk: `low_observed_review_bound_ambiguity`
- lab_result_records_with_treatment_or_medication_ambiguous_candidates: `3`
- lab_vs_imaging_risk: `low_observed_review_bound_ambiguity`
- lab_result_records_with_imaging_ambiguous_candidates: `3`
- lab_vs_generic_admin_table_risk: `no_direct_admin_or_insurance_cues_on_lab_predictions`
- lab_result_records_with_admin_or_insurance_cues: `0`
- stale_unknown_recomputation_risk: `controlled_by_lab_result_plus_latin_structure_cue_gate`
- classification_source_counts: `{'runtime_text_family_classifier': 444, 'not_recorded': 63}`
- unknown_ambiguous_candidate_sets: `{"('Imaging report', 'Medication plan', 'Treatment plan')": 7, "('Administrative / Insurance', 'Referral / Order')": 2, "('Imaging report', 'Medication plan')": 4, "('Administrative / Insurance', 'Medication plan', 'Referral / Order')": 1, "('Administrative / Insurance', 'Medication plan')": 1}`
- safety_conclusion: `No unsafe acceptance, external API use, or auto-accept allowance observed. Ambiguities remain review-bound.`

## Generalization Decision

- FAMILY-03 generalizes safely on this slice: `True`
- Follow-up classifier work justified: `not_immediate_for_lab_structure; investigate Unknown buckets only if operationally needed`
- Remaining Unknown recommendation: Leave Unknown/manual review as default. If coverage matters, prioritize text visibility and fallback-ran-no-family-match diagnostics before cue expansion.

## Validation Results

- `larger_slice_batch_eval`: passed: 507 evaluated; external_api_used_count 0; auto_accept_allowed_count 0; accepted_count 0
- `focused_family_eval_tests`: passed: 62 passed, 1 warning
- `ui_russian_ocr_upload_regression_group`: passed: 157 passed
- `python scripts/run_medai_ui_ops_panel_validation.py`: passed: medai_ui_ops_panel_ready
- `python scripts/run_medai_ui_boot_fix_validation.py`: passed: medai_ui_boot_fix_startup_resilience_ready
- `python scripts/run_cka_final_mvp_release_validation.py`: passed: 12/12 cases; 693 tests; external_api_used false
- `python scripts/run_b07_term01_opt_in_integration_validation.py`: passed: 6/6; external_api_used false
- `python scripts/run_medai_route_fix01_validation.py`: passed: medai_route_fix01_ready; external_api_used false
- `full_pytest`: not run; not practical after 507-file validation and prior 30-minute timeout on this workspace

## Safety / Privacy

- ocr_routing_changed: `False`
- ocr_engine_changed: `False`
- classifier_changed: `False`
- confidence_thresholds_changed: `False`
- auto_accept_expanded: `False`
- clinical_logic_changed: `False`
- lab_value_parsing_added: `False`
- medication_parsing_added: `False`
- dose_parsing_added: `False`
- ddi_logic_changed: `False`
- external_api_changed: `False`
- raw_ocr_text_in_public_reports: `False`
- raw_document_text_in_public_reports: `False`
- raw_filenames_private_paths_in_public_reports: `False`
- source_documents_staged: `False`
- affected_files_remain_review_bound: `True`

No raw OCR text, raw document text, raw filenames, private paths, PHI, or secrets are included.
