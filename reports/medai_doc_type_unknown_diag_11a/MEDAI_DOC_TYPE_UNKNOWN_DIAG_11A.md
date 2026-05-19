# MEDAI-DOC-TYPE-UNKNOWN-DIAG-11A - Latin Medical Abbreviation Spec

- branch: `clinical-knowledge-architecture`
- HEAD commit (short): `d5e7493ae628`
- source DIAG-10A commit (short): `fa4ac76`
- PARK-19 baseline commit (short): `ac466e0f9ab8`
- public_report_commit_hash_policy: `short_hashes_only`
- upstream:
  - `reports/medai_doc_type_unknown_diag_10a_language_propagation_operator_surface/(public)`
  - `reports/medai_doc_type_unknown_diag_09a_implementation/(public)`
  - `reports/medai_doc_type_unknown_diag_09a/(public spec)`
  - `reports/medai_doc_type_unknown_diag_08a_operator_badge_ui/(public)`
  - `reports/medai_doc_type_unknown_diag_07a_operator_routing_review/(public)`
  - `reports/medai_doc_type_unknown_diag_06a_implementation/(public)`
  - `reports/medai_doc_type_unknown_diag_06a/(public spec)`
  - `reports/medai_doc_type_unknown_diag_05/(public diagnostic)`
  - `reports/medai_doc_type_unknown_diag_04/(public diagnostic)`
- source report: `reports/medai_doc_type_family_04_larger_slice_validation/(public anonymized batch-eval per-file table)`
- total abbreviation-pool records analyzed: `8`
- overlap_with_numeric_table_safe_default_pool: `0`
- no_overlap_with_numeric_table_safe_default_pool: `True`
- overlap_with_language_propagation_pool: `0`
- no_overlap_with_language_propagation_pool: `True`
- suggested_future_env_var: `MEDAI_DOC_TYPE_LATIN_ABBREVIATION_METADATA_ENABLED`
- generated_at: `2026-05-19T09:35:14.431766+00:00`

## A. Required positive signal pattern

- `detector_attempted` = `yes`
- `detector_input_bucket` = `sufficient`
- `detector_confidence_bucket` = `high_or_medium`
- `script_detection_result` = `latin`
- `dominant_script` = `latin`
- `language_visibility_status` = `latin_visible_language_unknown`
- `latin_medical_abbrev_visible` = `yes`
- `medical_abbreviation_shape_detected` = `yes`
- `alphabetic_ratio_sufficient_for_language` = `yes`
- `no_cyrillic_dominant_signal` = `yes`
- `no_mixed_script_signal` = `yes`
- `no_low_confidence_detector_signal` = `yes`
- `not_already_handled_by_numeric_table_safe_default_helper` = `yes`
- `not_already_handled_by_language_propagation_helper` = `yes`

### Positive-signal match report on the priority slice

- `detector_attempted` expected=`yes`, matching=`8`, fully_matches=`True`
- `detector_input_bucket` expected=`sufficient`, matching=`8`, fully_matches=`True`
- `detector_confidence_bucket` expected=`high_or_medium`, matching=`8`, fully_matches=`True`
- `script_detection_result` expected=`latin`, matching=`8`, fully_matches=`True`
- `dominant_script` expected=`latin`, matching=`8`, fully_matches=`True`
- `language_visibility_status` expected=`latin_visible_language_unknown`, matching=`8`, fully_matches=`True`
- `latin_medical_abbrev_visible` expected=`yes`, matching=`8`, fully_matches=`True`
- `medical_abbreviation_shape_detected` expected=`yes`, matching=`8`, fully_matches=`True`
- `alphabetic_ratio_sufficient_for_language` expected=`yes`, matching=`8`, fully_matches=`True`
- `no_cyrillic_dominant_signal` expected=`yes`, matching=`8`, fully_matches=`True`
- `no_mixed_script_signal` expected=`yes`, matching=`8`, fully_matches=`True`
- `no_low_confidence_detector_signal` expected=`yes`, matching=`8`, fully_matches=`True`
- `not_already_handled_by_numeric_table_safe_default_helper` expected=`yes`, matching=`8`, fully_matches=`True`
- `not_already_handled_by_language_propagation_helper` expected=`yes`, matching=`8`, fully_matches=`True`

positive_signal_holds_on_all_priority_records: `True`

## B. Required exclusion rules

- `exclude_cyrillic_dominant_records`
- `exclude_mixed_script_records`
- `exclude_low_detector_confidence_records`
- `exclude_insufficient_detector_input_records`
- `exclude_no_text_layer_records`
- `exclude_image_like_but_not_routed_records`
- `exclude_table_heavy_numeric_safe_default_records_already_handled`
- `exclude_language_propagation_records_already_handled`
- `exclude_table_header_only_special_case_record`
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
- `exclude_language_propagation_records_already_handled` violating_record_count=`0`
- `exclude_table_header_only_special_case_record` violating_record_count=`0`
- `exclude_ambiguous_below_threshold_records` violating_record_count=`0`
- `exclude_fallback_ran_but_no_family_match_records` violating_record_count=`0`
- `exclude_medication_dose_or_ddi_interpretation` violating_record_count=`0`
- `exclude_lab_value_parsing` violating_record_count=`0`
- `exclude_records_with_insufficient_safe_metadata` violating_record_count=`0`

no_priority_record_violates_any_exclusion_rule: `True`

## C. Proposed future behavior (NOT implemented in this block)

- applies_to: `records matching the exact positive signal pattern only`
- proposed_label: `latin_medical_abbreviation_context`
- proposed_label_meaning: `Latin-script text contains medical-style abbreviations useful for language / context routing`
- default_action: `derive `latin_medical_abbreviation_context` as a safe metadata label only; never expand or parse the abbreviation`
- must_not_classify_clinical_meaning: `True`
- must_not_parse_the_abbreviation: `True`
- must_not_expand_the_abbreviation: `True`
- must_not_parse_values: `True`
- must_not_auto_accept: `True`
- must_not_write_active_clinical_facts: `True`
- must_keep_document_review_bound: `True`
- must_be_default_off_behind_separate_env_flag_or_operator_setting: `True`
- suggested_future_env_var: `MEDAI_DOC_TYPE_LATIN_ABBREVIATION_METADATA_ENABLED`
- must_not_overlap_with_numeric_table_safe_default_pool: `True`
- must_not_overlap_with_language_propagation_pool: `True`
- must_not_alter_raw_detector_output: `True`
- must_not_change_data_layer_document_type: `True`

## D. Future implementation acceptance criteria

- `only_exact_abbreviation_signature_records_receive_the_metadata_label`
- `no_overlap_with_numeric_table_safe_default_records`
- `no_overlap_with_language_propagation_records`
- `no_overlap_with_table_header_special_case_unless_explicitly_included_later`
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
- `replay_of_8_record_abbreviation_pool`
- `five_hundred_seven_file_aggregate_validation`
- `overlap_audit_against_diag_06a_helper_pool`
- `overlap_audit_against_diag_09a_helper_pool`
- `document_type_eval_regression_tests`
- `public_report_privacy_checks`
- `final_cka_mvp_validation`
- `b07_validation`
- `route_fix_validation`
- `ui_ops_validation`
- `ui_boot_validation`
- `staged_safety_check`

## Deferred subsets (out of scope)

- numeric_table_safe_default_pool_already_handled: 11 records handled by DIAG-06A/07A/08A; excluded from this spec
- language_propagation_pool_already_handled: 11 records handled by DIAG-09A/10A; excluded from this spec
- candidate_table_header_language_policy_record: 1 record from DIAG-05 routed to the table-header lever; deferred
- likely_text_layer_issue: 21 records deferred per DIAG-03
- fallback_ran_but_no_family_match: 17 records deferred per DIAG-02; no cue expansion
- ambiguous_below_threshold: 15 records excluded; review-bound, no cue expansion

## Implementation recommendation

- recommended_next: `future_block_named_unknown_diag_11a_implementation`

All 8 abbreviation-pool records match the exact positive signal pattern, no exclusion rule fires, and the slice is fully disjoint from both the numeric-table safe-default pool and the language-propagation pool. A future implementation block may prototype the abbreviation-context metadata helper inside the published acceptance criteria. The implementation must remain review-bound, default-off behind a separate env flag distinct from the two existing levers, and must never parse or expand the abbreviation itself.

## Progress estimate

- before_11a_unknown_track_done_pct: `approximately 87%`
- before_11a_unknown_track_remaining_pct: `approximately 13%`
- before_11a_project_done_pct: `approximately 82%`
- before_11a_project_remaining_pct: `approximately 18%`
- after_11a_unknown_track_done_pct: `approximately 90%`
- after_11a_unknown_track_remaining_pct: `approximately 10%`
- after_11a_project_done_pct: `approximately 83%`
- after_11a_project_remaining_pct: `approximately 17%`
- note: `Estimates are approximate and refer to the residual Unknown-reduction track in this workspace, plus the overall MedAI project state. They are informational only and not a release milestone.`

## Safety / Privacy

- behavior_changed: `False`
- external_api_used: `False`
- cue_expansion_recommended: `False`
- abbreviation_handling_implemented_in_this_block: `False`
- behavior_changed: `False`
- ocr_routing_changed: `False`
- ocr_engine_changed: `False`
- raw_language_detector_behavior_changed: `False`
- abbreviation_handling_behavior_changed: `False`
- classifier_behavior_changed: `False`
- thresholds_changed: `False`
- scoring_changed: `False`
- auto_accept_changed: `False`
- cue_packs_changed: `False`
- cue_expansion_recommended: `False`
- abbreviation_handling_implemented_in_this_block: `False`
- abbreviation_parsing_or_expansion_recommended: `False`
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
- no_overlap_with_language_propagation_pool: `True`

No raw filenames, raw OCR text, raw document text, private paths, PHI, or secrets are included. Specification-only block; no runtime behavior change, no abbreviation handling is implemented here, no cue expansion, no abbreviation parsing or expansion, no OCR routing or detector behavior change. Records remain review-bound.

## Raw signal distributions

- language_detector_attempted_counts: `yes`=8
- language_detector_input_bucket_counts: `sufficient`=8
- detector_confidence_bucket_counts: `high`=8
- script_detection_result_counts: `latin`=8
- dominant_script_counts: `latin`=8
- language_visibility_status_counts: `latin_visible_language_unknown`=8
- alphabetic_content_bucket_counts: `high`=8
- numeric_content_bucket_counts: `low`=4, `medium`=4
- symbol_content_bucket_counts: `high`=2, `medium`=6
- table_like_structure_detected_counts: `yes`=8
- section_heading_shape_detected_counts: `no`=6, `yes`=2
- medical_abbreviation_shape_detected_counts: `yes`=8
- lab_table_shape_detected_counts: `no`=4, `yes`=4
- imaging_modality_shape_detected_counts: `no`=3, `yes`=5
- image_like_pdf_counts: `no`=8
- pdf_text_layer_detected_counts: `yes`=8

## Why a spec block instead of an implementation

The 8 abbreviation-pool records share a uniform metadata signature 
(Latin script, high or medium detector confidence, sufficient 
input, visibility=latin_visible_language_unknown, medical_
abbreviation_shape_detected=yes) and are completely disjoint from 
both the numeric-table safe-default pool (11 records) and the 
language-propagation pool (11 records). Before any future 
implementation block touches runtime behavior, the exact positive 
signal pattern, exclusion rules, proposed default behavior, 
acceptance criteria, validation requirements, and rollback 
boundaries are published here in a single privacy-safe document. 
A future implementation block may proceed only inside the 
boundaries this spec defines.

## What this block did not change

- OCR routing logic
- OCR engine
- Raw language / script detector behavior
- Abbreviation handling behavior
- Classifier behavior
- Confidence thresholds or scoring
- Cue packs
- Auto-accept or review-bound rules
- B07 terminology, ROUTE-FIX, DB schema, command allowlist, external APIs

No abbreviation handling is implemented in this block. No clinical 
interpretation added. No values parsed. No abbreviation parsed or 
expanded. No active facts written.
