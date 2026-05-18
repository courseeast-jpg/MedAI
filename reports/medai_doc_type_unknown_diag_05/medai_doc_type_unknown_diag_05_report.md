# MEDAI-DOC-TYPE-UNKNOWN-DIAG-05 - Table-Heavy Language Detection Audit

- branch: `clinical-knowledge-architecture`
- HEAD commit (short): `e38704803e6f`
- PARK-19 baseline commit (short): `ac466e0f9ab8`
- public_report_commit_hash_policy: `short_hashes_only`
- upstream diagnostics:
  - `reports/medai_doc_type_unknown_diag_04/(public diagnostic)`
  - `reports/medai_doc_type_unknown_diag_03/(public diagnostic)`
  - `reports/medai_doc_type_unknown_diag_02/(public diagnostic)`
  - `reports/medai_doc_type_unknown_diag_01/(public diagnostic)`
- source report: `reports/medai_doc_type_family_04_larger_slice_validation/(public anonymized batch-eval per-file table)`
- target lever: `table_heavy_language_detection_policy_audit`
- total table-heavy lever records analyzed: `12`
- generated_at: `2026-05-18T22:20:22.643236+00:00`

## Table-structure evidence

- high_table_density: `12`
- medium_table_density: `0`
- low_table_density: `0`
- table_headers_visible: `1`
- repeated_row_pattern_visible: `12`
- insufficient_table_metadata: `0`

## Numeric-distribution evidence

- high_numeric_ratio: `0`
- medium_numeric_ratio: `12`
- low_numeric_ratio: `0`
- numeric_units_or_ranges_visible: `12`
- sparse_alpha_dense_numeric: `0`
- insufficient_numeric_metadata: `0`

## Alphabetic / script evidence

- latin_script_high_confidence: `12`
- latin_script_medium_confidence: `0`
- alphabetic_ratio_sufficient_for_language: `12`
- alphabetic_ratio_too_low_for_language: `0`
- dominant_script_confidence_missing: `0`
- insufficient_script_metadata: `0`

## Section / shape evidence

- lab_or_result_section_shape: `0`
- administrative_table_shape: `12`
- treatment_schedule_table_shape: `12`
- generic_table_shape: `0`
- no_section_shape_available: `0`

## Future-lever candidate counts (per-record single assignment)

- candidate_table_heavy_latin_policy: `0`
- candidate_table_header_language_policy: `1`
- candidate_numeric_table_safe_default_policy: `11`
- candidate_metadata_propagation_audit: `0`
- leave_manual_review: `0`
- insufficient_metadata_for_next_action: `0`

## Implementation-block recommendation

- implementation_block_justified: `True`
- choice: `C` (A=`candidate_table_heavy_latin_policy` prototype, B=`candidate_table_header_language_policy` prototype, C=`candidate_numeric_table_safe_default_policy` prototype, D=`candidate_metadata_propagation_audit` instead, E=`leave_manual_review`)

`candidate_numeric_table_safe_default_policy` is the only lever above the threshold (11 records). Recommend C.

## Deferred subsets (out of scope)

- language_detector_metadata_propagation_audit_pool: 11 records routed by DIAG-04 to this lever; deferred behind the table-heavy lever audit per the recommendation order
- latin_medical_abbreviation_handling_audit_pool: 8 records routed by DIAG-04 to this lever; deferred behind the table-heavy lever audit
- likely_text_layer_issue: 21 records deferred to the option-B follow-up per UNKNOWN-DIAG-03
- fallback_ran_but_no_family_match: 17 records deferred to a future cue-coverage audit per UNKNOWN-DIAG-02; no cue expansion
- ambiguous_below_threshold: 15 records remain excluded from optimization; review-bound, no cue expansion

## Safety / Privacy

- behavior_changed: `False`
- external_api_used: `False`
- cue_expansion_recommended: `False`
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
- metadata_propagation_audit_pool_deferred: `True`
- latin_medical_abbreviation_pool_deferred: `True`
- likely_text_layer_issue_deferred: `True`
- fallback_ran_but_no_family_match_deferred: `True`
- ambiguous_below_threshold_excluded: `True`

No raw filenames, raw OCR text, raw document text, private paths, PHI, or secrets are included. Diagnostic-only, evaluation/reporting changes only. This block recommends a follow-up diagnostic only; no policy is implemented here. Cue expansion is not recommended. No OCR routing or detector behavior change is proposed.
