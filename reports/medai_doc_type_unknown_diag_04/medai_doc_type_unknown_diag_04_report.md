# MEDAI-DOC-TYPE-UNKNOWN-DIAG-04 - Language/Script Detector Diagnostic

- branch: `clinical-knowledge-architecture`
- HEAD commit (short): `83531f679eff`
- PARK-19 baseline commit (short): `ac466e0f9ab8`
- public_report_commit_hash_policy: `short_hashes_only`
- upstream diagnostics:
  - `reports/medai_doc_type_unknown_diag_03/(public diagnostic)`
  - `reports/medai_doc_type_unknown_diag_02/(public diagnostic)`
  - `reports/medai_doc_type_unknown_diag_01/(public diagnostic)`
- source report: `reports/medai_doc_type_family_04_larger_slice_validation/(public anonymized batch-eval per-file table)`
- total language-detector records analyzed: `31`
- generated_at: `2026-05-18T18:35:18.214190+00:00`

## Sub-pool counts

- numeric_or_table_heavy_language_detector_gap: `16`
- latin_visible_language_unknown: `15`

## Evidence bucket counts

### numeric_or_table_heavy_language_detector_gap

- table_heavy_latin_visible: `16`
- numeric_heavy_latin_visible: `16`
- lab_table_shape_latin_visible: `2`
- sparse_words_many_numbers: `0`
- detector_input_too_structural: `5`
- insufficient_safe_metadata: `0`

### latin_visible_language_unknown

- latin_words_visible_detector_unknown: `15`
- latin_medical_abbrev_visible: `4`
- latin_table_headers_visible: `13`
- latin_script_detected_language_missing: `15`
- detector_output_not_propagated: `15`
- insufficient_safe_metadata: `0`

## Proposed future diagnostic lever counts

- language_detector_metadata_propagation_audit: `11`
- table_heavy_language_detection_policy_audit: `12`
- latin_medical_abbreviation_handling_audit: `8`
- leave_manual_review: `0`
- insufficient_metadata_for_next_action: `0`

## Implementation-block recommendation

- implementation_block_justified: `True`
- choice: `B` (A=`language_detector_metadata_propagation_audit`, B=`table_heavy_language_detection_policy_audit`, C=`latin_medical_abbreviation_handling_audit`, D=`leave_manual_review`)

Both `table_heavy_language_detection_policy_audit` (12) and `language_detector_metadata_propagation_audit` (11) meet the threshold. Recommend B first (larger pool, more direct lever), then A.

## Deferred subsets (out of scope)

- likely_text_layer_issue: 21 records deferred to the option B follow-up per UNKNOWN-DIAG-03
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
- likely_text_layer_issue_deferred: `True`
- fallback_ran_but_no_family_match_deferred: `True`
- ambiguous_below_threshold_excluded: `True`

No raw filenames, raw OCR text, raw document text, private paths, PHI, or secrets are included. Diagnostic-only, evaluation/reporting changes only. Cue expansion is not recommended for any bucket in this block. No OCR routing or detector behavior change is proposed in this block.
