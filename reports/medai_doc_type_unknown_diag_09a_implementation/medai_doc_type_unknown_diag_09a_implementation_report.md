# MEDAI-DOC-TYPE-UNKNOWN-DIAG-09A-IMPLEMENTATION - Language Detector Metadata Propagation

- branch: `clinical-knowledge-architecture`
- HEAD commit (short): `75bb1c4771b5`
- source spec commit (short): `5122d93`
- PARK-19 baseline commit (short): `ac466e0f9ab8`
- public_report_commit_hash_policy: `short_hashes_only`
- upstream:
  - `reports/medai_doc_type_unknown_diag_09a/(public spec)`
  - `reports/medai_doc_type_unknown_diag_08a_operator_badge_ui/(public)`
  - `reports/medai_doc_type_unknown_diag_07a_operator_routing_review/(public)`
  - `reports/medai_doc_type_unknown_diag_06a_implementation/(public)`
  - `reports/medai_doc_type_unknown_diag_06a/(public spec)`
  - `reports/medai_doc_type_unknown_diag_05/(public diagnostic)`
  - `reports/medai_doc_type_unknown_diag_04/(public diagnostic)`
- source report: `reports/medai_doc_type_family_04_larger_slice_validation/(public anonymized batch-eval per-file table)`
- propagated_metadata_label: `latin_detector_likely_english_context`
- propagated_metadata_disclaimer: `Safe metadata only. Not a final document type. Not clinical interpretation. Raw detector output unchanged.`
- language_propagation_metadata_env_var: `MEDAI_DOC_TYPE_LANGUAGE_PROPAGATION_METADATA_ENABLED`
- operator_review_badge_env_var_distinct: `MEDAI_DOC_TYPE_OPERATOR_REVIEW_BADGE_ENABLED`
- flag_separation_confirmed: `True`
- helper_default_disabled: `True`
- generated_at: `2026-05-19T03:58:18.441637+00:00`

## Implementation summary

Adds `clinical_knowledge.document_type.derive_language_propagation_metadata_label` as a pure default-off helper. Returns `latin_detector_likely_english_context` only when (a) the helper is explicitly enabled via `enabled=True` OR the SEPARATE env var `MEDAI_DOC_TYPE_LANGUAGE_PROPAGATION_METADATA_ENABLED` is truthy, (b) every field of the 11-field positive signal pattern holds, (c) none of the 12 exclusion rules fires, and (d) none of 5 implementation-level safeguards fires (must_be_predicted_document_type_unknown, must_be_in_insufficient_text_visibility_bucket, must_be_in_language_visibility_unknown_routing_bucket, must_have_no_medical_abbreviation_shape_detected, must_not_be_in_table_heavy_diag03_sub_pool). The safeguards close the same kind of gap that DIAG-06A-IMPLEMENTATION closed: on the full 507-row corpus the bare 11-field signature matches many records DIAG-02 / DIAG-03 / DIAG-04 route to other levers; the safeguards ensure the helper labels only the exact 11 propagation-pool records. The helper is pure, never mutates the record, never modifies raw detector output, never auto-accepts, never changes the data-layer document type, never classifies clinical meaning, never parses lab values / medications / doses / DDIs, and never writes active clinical facts.

## Rollback / disable path

Default-off. Any one of the following is sufficient to disable: (1) omit the `enabled` kwarg AND leave the SEPARATE env var `MEDAI_DOC_TYPE_LANGUAGE_PROPAGATION_METADATA_ENABLED` unset; (2) set the env var to a falsy value (`0` / `false` / `no` / `off` / `disabled`); (3) pass `enabled=False` explicitly (overrides any env setting); (4) never import the module - existing pipelines are unaffected. Note: the DIAG-07A operator-badge env var `MEDAI_DOC_TYPE_OPERATOR_REVIEW_BADGE_ENABLED` does NOT enable this helper - the env vars are deliberately separate so each can be rolled back independently.

## Positive signal pattern

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

## Exclusion rules

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

## 11-record propagation-pool replay

- priority_slice_size: `11`
- enabled_labeled_count: `11`
- disabled_labeled_count: `0`
- default_off_labeled_count: `0`
- matches_priority_slice_exactly: `True`

## 507-file aggregate

- corpus_size: `507`
- default_off_labeled_count: `0`
- explicit_enabled_labeled_count: `11`
- propagation_env_enabled_labeled_count: `11`
- operator_review_env_only_labeled_count: `0`
- extras_outside_priority_count: `0`
- missing_from_priority_count: `0`
- no_false_positive_outside_priority: `True`
- no_false_negative_inside_priority: `True`
- flag_separation_holds: `True`

## Overlap with numeric-table safe-default pool

- overlap_with_numeric_table_safe_default_pool: `0`
- no_overlap_with_numeric_table_safe_default_pool: `True`

## Counts

- unknown_count_before: `107`
- unknown_count_after: `107`
- unknown_count_impact_delta: `0`
- accepted_count: `0`
- auto_accept_allowed_count: `0`
- external_api_used_count: `0`

## Review-bound preservation

- review_bound_records_before: `507`
- review_bound_records_after: `507`
- review_bound_preserved: `True`

## False-positive audit

- numeric_table_safe_default_overlap: `0`
- treatment_or_schedule_expansion: `0`
- imaging_expansion: `0`
- administrative_or_table_expansion: `0`
- other_expansion: `0`
- no_false_positive_expansion: `True`

## Deferred subsets (out of scope)

- numeric_table_safe_default_pool_already_handled: 11 records covered by DIAG-06A/07A/08A; excluded from this helper
- candidate_latin_medical_abbreviation_handling_audit_pool: 8 records from DIAG-04 routed to the abbreviation lever; deferred
- candidate_table_header_language_policy_record: 1 record from DIAG-05 routed to the table-header lever; deferred
- likely_text_layer_issue: 21 records deferred per DIAG-03
- fallback_ran_but_no_family_match: 17 records deferred per DIAG-02; no cue expansion
- ambiguous_below_threshold: 15 records excluded; review-bound, no cue expansion

## Progress estimate

- before_impl_unknown_track_done_pct: `approximately 80%`
- before_impl_unknown_track_remaining_pct: `approximately 20%`
- before_impl_project_done_pct: `approximately 80%`
- before_impl_project_remaining_pct: `approximately 20%`
- after_impl_unknown_track_done_pct: `approximately 84%`
- after_impl_unknown_track_remaining_pct: `approximately 16%`
- after_impl_project_done_pct: `approximately 81%`
- after_impl_project_remaining_pct: `approximately 19%`
- note: `Estimates are approximate and refer to the residual Unknown-reduction track in this workspace, plus the overall MedAI project state. They are informational only and not a release milestone.`

## Safety / Privacy

- behavior_changed: `True` (strictly limited to safe metadata propagation helper)
- behavior_change_scope: Strictly limited to deriving the safe propagated metadata label `latin_detector_likely_english_context` for records that match the exact 11-field positive signal pattern AND satisfy 5 implementation-level safeguards AND show zero overlap with the numeric-table safe-default pool AND only when the SEPARATE propagation env var is explicitly enabled. No clinical interpretation, no value parsing, no auto-accept, no active clinical fact writes, no document-type promotion at the data layer, no OCR routing or detector behavior change.
- clinical_behavior_changed: `False`
- external_api_used: `False`
- cue_expansion_recommended: `False`
- behavior_changed_strictly_limited_to_safe_metadata_propagation: `True`
- raw_detector_output_unchanged: `True`
- clinical_behavior_changed: `False`
- ocr_routing_changed: `False`
- ocr_engine_changed: `False`
- raw_language_detector_behavior_changed: `False`
- classifier_behavior_changed_for_non_signature_records: `False`
- data_layer_document_type_changed: `False`
- thresholds_changed: `False`
- scoring_changed: `False`
- auto_accept_changed: `False`
- cue_packs_changed: `False`
- cue_expansion_recommended: `False`
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
- helper_default_disabled: `True`
- rollback_path_present: `True`
- propagation_env_var_separate_from_operator_badge_env_var: `True`
- no_overlap_with_numeric_table_safe_default_pool: `True`

No raw filenames, raw OCR text, raw document text, private paths, PHI, or secrets are included. The runtime behavior change is strictly limited to the safe metadata propagation helper described above; no clinical interpretation, no value parsing, no auto-accept, no active clinical fact writes, no document-type promotion at the data layer. Review-bound status is preserved. Raw detector output is unchanged.
