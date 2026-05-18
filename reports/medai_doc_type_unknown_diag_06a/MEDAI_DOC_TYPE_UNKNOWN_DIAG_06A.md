# MEDAI-DOC-TYPE-UNKNOWN-DIAG-06A - Numeric-Table Safe-Default Spec

- branch: `clinical-knowledge-architecture`
- HEAD commit (short): `37536a5e79ad`
- PARK-19 baseline commit (short): `ac466e0f9ab8`
- public_report_commit_hash_policy: `short_hashes_only`
- upstream diagnostics:
  - `reports/medai_doc_type_unknown_diag_05/(public diagnostic)`
  - `reports/medai_doc_type_unknown_diag_04/(public diagnostic)`
  - `reports/medai_doc_type_unknown_diag_03/(public diagnostic)`
  - `reports/medai_doc_type_unknown_diag_02/(public diagnostic)`
  - `reports/medai_doc_type_unknown_diag_01/(public diagnostic)`
- source report: `reports/medai_doc_type_family_04_larger_slice_validation/(public anonymized batch-eval per-file table)`
- total numeric-table records analyzed: `11`
- deferred table-header record count: `1`
- generated_at: `2026-05-18T22:34:41.398071+00:00`

## A. Required positive signature

- `table_like_structure_detected` = `yes`
- `high_table_density_required` = `yes`
- `repeated_row_pattern_visible` = `yes`
- `numeric_content_bucket` = `medium`
- `numeric_units_or_ranges_visible` = `yes`
- `script_detection_result` = `latin`
- `dominant_script` = `latin`
- `detector_confidence_bucket` = `high`
- `alphabetic_ratio_sufficient_for_language` = `yes`
- `administrative_table_shape` = `yes`
- `treatment_schedule_table_shape` = `yes`
- `language_detector_attempted` = `yes`
- `language_detector_input_bucket` = `sufficient`
- `language_visibility_status` = `latin_visible_language_unknown`

### Positive-signature match report on the priority slice

- `table_like_structure_detected` expected=`yes`, matching=`11`, fully_matches=`True`
- `high_table_density_required` expected=`yes`, matching=`11`, fully_matches=`True`
- `repeated_row_pattern_visible` expected=`yes`, matching=`11`, fully_matches=`True`
- `numeric_content_bucket` expected=`medium`, matching=`11`, fully_matches=`True`
- `numeric_units_or_ranges_visible` expected=`yes`, matching=`11`, fully_matches=`True`
- `script_detection_result` expected=`latin`, matching=`11`, fully_matches=`True`
- `dominant_script` expected=`latin`, matching=`11`, fully_matches=`True`
- `detector_confidence_bucket` expected=`high`, matching=`11`, fully_matches=`True`
- `alphabetic_ratio_sufficient_for_language` expected=`yes`, matching=`11`, fully_matches=`True`
- `administrative_table_shape` expected=`yes`, matching=`11`, fully_matches=`True`
- `treatment_schedule_table_shape` expected=`yes`, matching=`11`, fully_matches=`True`
- `language_detector_attempted` expected=`yes`, matching=`11`, fully_matches=`True`
- `language_detector_input_bucket` expected=`sufficient`, matching=`11`, fully_matches=`True`
- `language_visibility_status` expected=`latin_visible_language_unknown`, matching=`11`, fully_matches=`True`

positive_signature_holds_on_all_priority_records: `True`

## B. Required exclusion rules

- `exclude_cyrillic_dominant_records`
- `exclude_mixed_script_records`
- `exclude_low_alphabetic_ratio_records`
- `exclude_no_text_layer_records`
- `exclude_image_like_but_not_routed_records`
- `exclude_ambiguous_below_threshold_records`
- `exclude_fallback_ran_but_no_family_match_records`
- `exclude_medication_dose_or_ddi_interpretation`
- `exclude_lab_value_parsing`
- `exclude_records_with_insufficient_safe_metadata`

### Exclusion-rule audit on the priority slice

- `exclude_cyrillic_dominant_records` violating_record_count=`0`
- `exclude_mixed_script_records` violating_record_count=`0`
- `exclude_low_alphabetic_ratio_records` violating_record_count=`0`
- `exclude_no_text_layer_records` violating_record_count=`0`
- `exclude_image_like_but_not_routed_records` violating_record_count=`0`
- `exclude_ambiguous_below_threshold_records` violating_record_count=`0`
- `exclude_fallback_ran_but_no_family_match_records` violating_record_count=`0`
- `exclude_medication_dose_or_ddi_interpretation` violating_record_count=`0`
- `exclude_lab_value_parsing` violating_record_count=`0`
- `exclude_records_with_insufficient_safe_metadata` violating_record_count=`0`

no_priority_record_violates_any_exclusion_rule: `True`

## C. Proposed future default (NOT implemented in this block)

- applies_to: `records matching the exact positive signature only`
- default_action: `assign language visibility as latin_script_likely_english_table_context for routing and metadata purposes only`
- scope_of_effect: `routing and metadata propagation only; no classifier outcome change, no clinical interpretation, no value parsing`
- must_not_auto_accept: `True`
- must_not_classify_clinical_meaning: `True`
- must_not_parse_values: `True`
- must_not_write_active_clinical_facts: `True`
- must_keep_document_review_bound: `True`

## D. Future implementation acceptance criteria

- `unknown_count_decreases_only_for_exact_signature_records`
- `accepted_count_remains_zero`
- `auto_accept_allowed_count_remains_zero`
- `external_api_used_count_remains_zero`
- `all_affected_records_remain_review_bound`
- `no_new_treatment_imaging_or_admin_false_positive_expansion`
- `public_report_privacy_checks_remain_clean`
- `rollback_flag_or_disable_path_exists`

## E. Future validation requirements

- `focused_synthetic_tests`
- `replay_of_11_record_anonymous_subset`
- `larger_507_file_aggregate_validation`
- `document_type_eval_regression_tests`
- `public_report_privacy_checks`
- `final_cka_mvp_validation`
- `b07_validation`
- `route_fix_validation`
- `staged_safety_check`

## Deferred subsets (out of scope)

- candidate_table_header_language_policy_record_count: 1 record(s); deferred as a special case to be absorbed into a future numeric-table spec implementation
- language_detector_metadata_propagation_audit_pool: 11 records from DIAG-04 routed to that lever; deferred
- latin_medical_abbreviation_handling_audit_pool: 8 records from DIAG-04 routed to that lever; deferred
- likely_text_layer_issue: 21 records deferred per DIAG-03
- fallback_ran_but_no_family_match: 17 records deferred per DIAG-02; no cue expansion
- ambiguous_below_threshold: 15 records excluded; review-bound, no cue expansion

## Implementation recommendation

- recommended_next: `future_block_named_unknown_diag_06a_implementation`

All 11 priority records match the exact positive signature and none violate any exclusion rule. A future implementation block may prototype the numeric-table safe-default safe-default language lever inside the published acceptance criteria for that lever. The implementation must remain review-bound and must not change clinical interpretation, OCR routing, language detector behavior, or cue packs.

## Progress estimate

- before_06a_unknown_track_done_pct: `approximately 55%`
- before_06a_unknown_track_remaining_pct: `approximately 45%`
- after_06a_unknown_track_done_pct: `approximately 62%`
- after_06a_unknown_track_remaining_pct: `approximately 38%`
- after_06a_project_done_pct: `approximately 76%`
- after_06a_project_remaining_pct: `approximately 24%`
- note: `Estimates are approximate and refer to the residual Unknown-reduction track in this workspace, plus the overall MedAI project state. They are informational only and not a release milestone.`

## Safety / Privacy

- behavior_changed: `False`
- external_api_used: `False`
- cue_expansion_recommended: `False`
- policy_implemented_in_this_block: `False`
- behavior_changed: `False`
- ocr_routing_changed: `False`
- ocr_engine_changed: `False`
- language_detector_behavior_changed: `False`
- classifier_behavior_changed: `False`
- thresholds_changed: `False`
- scoring_changed: `False`
- auto_accept_changed: `False`
- cue_packs_changed: `False`
- cue_expansion_recommended: `False`
- policy_implemented_in_this_block: `False`
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
- deferred_pools_remain_deferred: `True`

No raw filenames, raw OCR text, raw document text, private paths, PHI, or secrets are included. Specification-only block; no runtime behavior change, no policy is implemented here, no cue expansion, no OCR routing or detector behavior change.

## Raw signal distributions

- dominant_script_counts: `latin`=11
- language_visibility_status_counts: `latin_visible_language_unknown`=11
- detector_confidence_bucket_counts: `high`=11
- table_like_structure_detected_counts: `yes`=11
- numeric_content_bucket_counts: `medium`=11
- alphabetic_content_bucket_counts: `high`=11
- symbol_content_bucket_counts: `medium`=11
- administrative_form_shape_detected_counts: `yes`=11
- date_or_schedule_shape_detected_counts: `yes`=11
- section_heading_shape_detected_counts: `no`=11
- lab_table_shape_detected_counts: `no`=11
- imaging_modality_shape_detected_counts: `no`=11
- medical_abbreviation_shape_detected_counts: `no`=11
- image_like_pdf_counts: `no`=11
- pdf_text_layer_detected_counts: `yes`=11

## Why a spec block instead of an implementation

The 11 priority records share an unusually uniform metadata 
signature. Before any future implementation block touches runtime 
behavior, the exact positive signature, exclusion rules, default 
action, acceptance criteria, and validation requirements are 
published here in a single privacy-safe document. A future 
implementation block may then proceed only if it stays inside the 
boundaries this spec defines and inside the acceptance criteria 
this spec lists.

## What this block did not change

- OCR routing logic
- OCR engine
- Language / script detector behavior
- Classifier behavior or cue packs
- Confidence thresholds or scoring
- Auto-accept / review-bound policy
- B07 terminology, ROUTE-FIX, DB schema, command allowlist, external APIs

No policy is implemented in this block. No clinical interpretation 
added. No values parsed. No active facts written.
