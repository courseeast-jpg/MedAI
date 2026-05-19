# MEDAI-DOC-TYPE-UNKNOWN-DIAG-09A - Language Detector Propagation Spec

- branch: `clinical-knowledge-architecture`
- HEAD commit (short): `9d7c4ec33102`
- source DIAG-08A commit (short): `f5b80ce`
- PARK-19 baseline commit (short): `ac466e0f9ab8`
- public_report_commit_hash_policy: `short_hashes_only`
- upstream diagnostics / spec:
  - `reports/medai_doc_type_unknown_diag_08a_operator_badge_ui/(public)`
  - `reports/medai_doc_type_unknown_diag_07a_operator_routing_review/(public)`
  - `reports/medai_doc_type_unknown_diag_06a_implementation/(public)`
  - `reports/medai_doc_type_unknown_diag_06a/(public spec)`
  - `reports/medai_doc_type_unknown_diag_05/(public diagnostic)`
  - `reports/medai_doc_type_unknown_diag_04/(public diagnostic)`
  - `reports/medai_doc_type_unknown_diag_03/(public diagnostic)`
  - `reports/medai_doc_type_unknown_diag_02/(public diagnostic)`
  - `reports/medai_doc_type_unknown_diag_01/(public diagnostic)`
- source report: `reports/medai_doc_type_family_04_larger_slice_validation/(public anonymized batch-eval per-file table)`
- total propagation-pool records analyzed: `11`
- overlap_with_numeric_table_safe_default_pool: `0`
- no_overlap_with_numeric_table_safe_default_pool: `True`
- generated_at: `2026-05-19T03:47:16.307850+00:00`

## A. Required positive signal pattern

- `detector_attempted` = `yes`
- `detector_input_bucket` = `sufficient`
- `detector_confidence_bucket` = `high`
- `script_detection_result` = `latin`
- `dominant_script` = `latin`
- `language_visibility_status` = `latin_visible_language_unknown`
- `detector_output_not_propagated` = `yes`
- `alphabetic_ratio_sufficient_for_language` = `yes`
- `no_cyrillic_dominant_signal` = `yes`
- `no_mixed_script_signal` = `yes`
- `no_low_confidence_detector_signal` = `yes`

### Positive-signal match report on the priority slice

- `detector_attempted` expected=`yes`, matching=`11`, fully_matches=`True`
- `detector_input_bucket` expected=`sufficient`, matching=`11`, fully_matches=`True`
- `detector_confidence_bucket` expected=`high`, matching=`11`, fully_matches=`True`
- `script_detection_result` expected=`latin`, matching=`11`, fully_matches=`True`
- `dominant_script` expected=`latin`, matching=`11`, fully_matches=`True`
- `language_visibility_status` expected=`latin_visible_language_unknown`, matching=`11`, fully_matches=`True`
- `detector_output_not_propagated` expected=`yes`, matching=`11`, fully_matches=`True`
- `alphabetic_ratio_sufficient_for_language` expected=`yes`, matching=`11`, fully_matches=`True`
- `no_cyrillic_dominant_signal` expected=`yes`, matching=`11`, fully_matches=`True`
- `no_mixed_script_signal` expected=`yes`, matching=`11`, fully_matches=`True`
- `no_low_confidence_detector_signal` expected=`yes`, matching=`11`, fully_matches=`True`

positive_signal_holds_on_all_priority_records: `True`

## B. Required exclusion rules

- `exclude_cyrillic_dominant_records`
- `exclude_mixed_script_records`
- `exclude_low_detector_confidence_records`
- `exclude_insufficient_detector_input_records`
- `exclude_no_text_layer_records`
- `exclude_image_like_but_not_routed_records`
- `exclude_table_heavy_numeric_safe_default_records_already_handled`
- `exclude_ambiguous_below_threshold_records`
- `exclude_fallback_ran_but_no_family_match_records`
- `exclude_medication_dose_or_ddi_interpretation`
- `exclude_lab_value_parsing`
- `exclude_records_with_insufficient_safe_metadata`

### Exclusion-rule audit on the priority slice

- `exclude_cyrillic_dominant_records` violating_record_count=`0`
- `exclude_mixed_script_records` violating_record_count=`0`
- `exclude_low_detector_confidence_records` violating_record_count=`0`
- `exclude_insufficient_detector_input_records` violating_record_count=`0`
- `exclude_no_text_layer_records` violating_record_count=`0`
- `exclude_image_like_but_not_routed_records` violating_record_count=`0`
- `exclude_table_heavy_numeric_safe_default_records_already_handled` violating_record_count=`0`
- `exclude_ambiguous_below_threshold_records` violating_record_count=`0`
- `exclude_fallback_ran_but_no_family_match_records` violating_record_count=`0`
- `exclude_medication_dose_or_ddi_interpretation` violating_record_count=`0`
- `exclude_lab_value_parsing` violating_record_count=`0`
- `exclude_records_with_insufficient_safe_metadata` violating_record_count=`0`

no_priority_record_violates_any_exclusion_rule: `True`

## C. Proposed future propagation (NOT implemented in this block)

- applies_to: `records matching the exact positive signal pattern only`
- default_action: `propagate detector-side latin/likely-english context into safe metadata only; record the propagation as a derived field alongside the raw detector output, never replacing it`
- scope_of_effect: `metadata propagation only; raw detector output is untouched, classifier outcome is unchanged, no clinical interpretation, no value parsing`
- must_not_alter_raw_detector_output: `True`
- must_not_auto_accept: `True`
- must_not_classify_clinical_meaning: `True`
- must_not_parse_values: `True`
- must_not_write_active_clinical_facts: `True`
- must_keep_document_review_bound: `True`
- must_be_default_off_behind_separate_env_flag_or_operator_setting: `True`
- must_not_overlap_with_numeric_table_safe_default_pool: `True`

## D. Future implementation acceptance criteria

- `only_exact_propagation_signature_records_receive_propagated_metadata`
- `no_overlap_with_numeric_table_safe_default_records`
- `accepted_count_remains_zero`
- `auto_accept_allowed_count_remains_zero`
- `external_api_used_count_remains_zero`
- `all_affected_records_remain_review_bound`
- `data_layer_unknown_count_behavior_explicitly_reported`
- `no_treatment_imaging_or_admin_false_positive_expansion`
- `public_report_privacy_checks_remain_clean`
- `rollback_or_disable_path_exists`

## E. Future validation requirements

- `focused_synthetic_tests`
- `replay_of_11_record_propagation_pool`
- `five_hundred_seven_file_aggregate_validation`
- `document_type_eval_regression_tests`
- `public_report_privacy_checks`
- `final_cka_mvp_validation`
- `b07_validation`
- `route_fix_validation`
- `ui_ops_validation`
- `ui_boot_validation`
- `staged_safety_check`

## Deferred subsets (out of scope)

- numeric_table_safe_default_pool_already_handled: 11 records covered by DIAG-06A/07A/08A; excluded from this spec
- candidate_latin_medical_abbreviation_handling_audit_pool: 8 records from DIAG-04 routed to the abbreviation lever; deferred
- candidate_table_header_language_policy_record: 1 record from DIAG-05 routed to the table-header lever; deferred
- likely_text_layer_issue: 21 records deferred per DIAG-03
- fallback_ran_but_no_family_match: 17 records deferred per DIAG-02; no cue expansion
- ambiguous_below_threshold: 15 records excluded; review-bound, no cue expansion

## Implementation recommendation

- recommended_next: `future_block_named_unknown_diag_09a_implementation`

All 11 propagation-pool records match the exact positive signal pattern, no exclusion rule fires, and there is zero overlap with the numeric-table safe-default pool. A future implementation block may prototype the metadata propagation behavior inside the published acceptance criteria for that future block. The implementation must remain review-bound, default-off behind a separate env flag or operator setting, and must not alter raw detector output, classifier behavior, OCR routing, or any clinical interpretation.

## Progress estimate

- before_09a_unknown_track_done_pct: `approximately 76%`
- before_09a_unknown_track_remaining_pct: `approximately 24%`
- before_09a_project_done_pct: `approximately 79%`
- before_09a_project_remaining_pct: `approximately 21%`
- after_09a_unknown_track_done_pct: `approximately 80%`
- after_09a_unknown_track_remaining_pct: `approximately 20%`
- after_09a_project_done_pct: `approximately 80%`
- after_09a_project_remaining_pct: `approximately 20%`
- note: `Estimates are approximate and refer to the residual Unknown-reduction track in this workspace, plus the overall MedAI project state. They are informational only and not a release milestone.`

## Safety / Privacy

- behavior_changed: `False`
- external_api_used: `False`
- cue_expansion_recommended: `False`
- propagation_implemented_in_this_block: `False`
- behavior_changed: `False`
- ocr_routing_changed: `False`
- ocr_engine_changed: `False`
- raw_language_detector_behavior_changed: `False`
- language_metadata_propagation_behavior_changed: `False`
- classifier_behavior_changed: `False`
- thresholds_changed: `False`
- scoring_changed: `False`
- auto_accept_changed: `False`
- cue_packs_changed: `False`
- cue_expansion_recommended: `False`
- propagation_implemented_in_this_block: `False`
- lab_value_parsing_added: `False`
- medication_parsing_added: `False`
- dose_parsing_added: `False`
- ddi_logic_changed: `False`
- clinical_interpretation_added: `False`
- b07_changed: `False`
- route_fix_changed: `False`
- db_schema_changed: `False`
- command_allowlist_changed: `False`
- external_api_changed: `False`
- external_api_used: `False`
- raw_filenames_in_public_reports: `False`
- raw_ocr_text_in_public_reports: `False`
- raw_document_text_in_public_reports: `False`
- private_paths_in_public_reports: `False`
- source_documents_staged: `False`
- private_corpus_files_staged: `False`
- secrets_in_public_reports: `False`
- all_records_remain_review_bound: `True`
- no_overlap_with_numeric_table_safe_default_pool: `True`

No raw filenames, raw OCR text, raw document text, private paths, PHI, or secrets are included. Specification-only block; no runtime behavior change, no propagation behavior is implemented here, no cue expansion, no OCR routing or detector behavior change. Records remain review-bound.

## Raw signal distributions

- language_detector_attempted_counts: `yes`=11
- language_detector_input_bucket_counts: `sufficient`=11
- detector_confidence_bucket_counts: `high`=11
- script_detection_result_counts: `latin`=11
- dominant_script_counts: `latin`=11
- language_visibility_status_counts: `latin_visible_language_unknown`=11
- alphabetic_content_bucket_counts: `high`=11
- numeric_content_bucket_counts: `low`=11
- symbol_content_bucket_counts: `high`=1, `medium`=10
- table_like_structure_detected_counts: `no`=2, `yes`=9
- section_heading_shape_detected_counts: `no`=11
- medical_abbreviation_shape_detected_counts: `no`=11
- lab_table_shape_detected_counts: `no`=10, `yes`=1
- imaging_modality_shape_detected_counts: `no`=11
- language_script_detector_unknown_bucket_counts: `detector_input_garbled_or_mojibake`=2, `detector_input_symbol_heavy`=1, `script_detectable_language_unknown`=8
- image_like_pdf_counts: `no`=11
- pdf_text_layer_detected_counts: `yes`=11

## Why a spec block instead of an implementation

The 11 propagation-pool records share a uniform metadata signature 
(high-confidence Latin detector input, sufficient input bucket, 
visibility flagged as latin_visible_language_unknown) and are 
completely disjoint from the 11 numeric-table safe-default records 
already handled by DIAG-06A/07A/08A. Before any future 
implementation block touches runtime behavior, the exact positive 
signal pattern, exclusion rules, proposed propagation behavior, 
acceptance criteria, and validation requirements are published 
here in a single privacy-safe document. A future implementation 
block may proceed only inside the boundaries this spec defines.

## What this block did not change

- OCR routing logic
- OCR engine
- Raw language / script detector behavior
- Language metadata propagation behavior
- Classifier behavior
- Confidence thresholds or scoring
- Cue packs
- Auto-accept or review-bound rules
- B07 terminology, ROUTE-FIX, DB schema, command allowlist, external APIs

No propagation behavior is implemented in this block. No clinical 
interpretation. No values parsed. No active facts written.
