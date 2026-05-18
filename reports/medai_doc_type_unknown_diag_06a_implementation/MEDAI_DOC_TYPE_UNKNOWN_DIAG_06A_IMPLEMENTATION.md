# MEDAI-DOC-TYPE-UNKNOWN-DIAG-06A-IMPLEMENTATION - Numeric-Table Safe-Default Label

- branch: `clinical-knowledge-architecture`
- HEAD commit (short): `d4eeb053bf99`
- source spec commit (short): `70a2b59`
- PARK-19 baseline commit (short): `ac466e0f9ab8`
- public_report_commit_hash_policy: `short_hashes_only`
- upstream diagnostics / spec:
  - `reports/medai_doc_type_unknown_diag_06a/(public spec)`
  - `reports/medai_doc_type_unknown_diag_05/(public diagnostic)`
  - `reports/medai_doc_type_unknown_diag_04/(public diagnostic)`
  - `reports/medai_doc_type_unknown_diag_03/(public diagnostic)`
  - `reports/medai_doc_type_unknown_diag_02/(public diagnostic)`
  - `reports/medai_doc_type_unknown_diag_01/(public diagnostic)`
- source report: `reports/medai_doc_type_family_04_larger_slice_validation/(public anonymized batch-eval per-file table)`
- derived label: `latin_script_likely_english_table_context`
- helper_default_disabled: `True`
- generated_at: `2026-05-18T23:00:09.563973+00:00`

## Implementation summary

Adds a small pure helper `clinical_knowledge.document_type.derive_numeric_table_safe_default_label` that returns `latin_script_likely_english_table_context` ONLY when (a) the caller explicitly passes `enabled=True`, (b) the record matches every field of the 14-field positive signature, (c) none of the 10 exclusion rules fires, and (d) none of the implementation-level safeguards fires. The helper is pure, default-off, and never mutates the record. The disable / rollback path is `enabled=False` (or simply not calling the helper); the function returns `None` in that case and existing pipelines that never import the module are unaffected. Raw language-detector output is not modified; the label is added as safe metadata only. Auto-accept, clinical interpretation, lab-value parsing, medication / dose / DDI parsing, and active clinical fact writes are all forbidden by the helper's contract.

## Rollback / disable path

Pass `enabled=False` (the default) or stop importing the module. No persisted state to roll back. The helper is additive and pure.

## Positive signature

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

## Exclusion rules

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

## 11-record replay

- priority_slice_size: `11`
- enabled_labeled_count: `11`
- disabled_labeled_count: `0`
- matches_priority_slice_exactly: `True`

## 507-file aggregate

- corpus_size: `507`
- enabled_labeled_count: `11`
- disabled_labeled_count: `0`
- extras_outside_priority_slice: `[]`
- extras_count: `0`
- missing_from_priority_count: `0`
- no_false_positive_outside_priority: `True`
- no_false_negative_inside_priority: `True`

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

- treatment_expansion: `0`
- imaging_expansion: `0`
- administrative_expansion: `0`
- other_expansion: `0`
- no_false_positive_expansion: `True`

## Deferred subsets (out of scope)

- candidate_table_header_language_policy_record: 1 record from DIAG-05 routed to the table-header lever; deferred
- candidate_metadata_propagation_audit_pool: 11 records from DIAG-04 routed to the propagation-audit lever; deferred
- candidate_latin_medical_abbreviation_handling_audit_pool: 8 records from DIAG-04 routed to the abbreviation lever; deferred
- likely_text_layer_issue: 21 records deferred per DIAG-03
- fallback_ran_but_no_family_match: 17 records deferred per DIAG-02; no cue expansion
- ambiguous_below_threshold: 15 records excluded; review-bound, no cue expansion

## Progress estimate

- before_impl_unknown_track_done_pct: `approximately 62%`
- before_impl_unknown_track_remaining_pct: `approximately 38%`
- before_impl_project_done_pct: `approximately 76%`
- before_impl_project_remaining_pct: `approximately 24%`
- after_impl_unknown_track_done_pct: `approximately 68%`
- after_impl_unknown_track_remaining_pct: `approximately 32%`
- after_impl_project_done_pct: `approximately 77%`
- after_impl_project_remaining_pct: `approximately 23%`
- note: `Estimates are approximate and refer to the residual Unknown-reduction track in this workspace, plus the overall MedAI project state. They are informational only and not a release milestone.`

## Safety / Privacy

- behavior_changed: `True` (strictly limited to safe metadata / routing label only)
- clinical_behavior_changed: `False`
- external_api_used: `False`
- cue_expansion_recommended: `False`
- policy_implementation_scope: Strictly limited to deriving a safe metadata / routing label for records that match the exact 14-field positive signature and violate none of the 10 exclusion rules. No clinical interpretation, no value parsing, no auto-accept, no active clinical fact writes. Review-bound status preserved. Default-off; callers must explicitly opt in via `enabled=True`.
- behavior_changed_strictly_limited_to_safe_metadata_label: `True`
- clinical_behavior_changed: `False`
- ocr_routing_changed: `False`
- ocr_engine_changed: `False`
- language_detector_behavior_changed: `False`
- classifier_behavior_changed_for_non_signature_records: `False`
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

No raw filenames, raw OCR text, raw document text, private paths, PHI, or secrets are included. The runtime behavior change is strictly limited to the safe metadata / routing label described above; no clinical interpretation, no value parsing, no auto-accept, no active clinical fact writes. Review-bound status is preserved.

## Recommendation for next block

The numeric-table safe-default metadata label is now available behind a default-off helper. Recommend a downstream evaluation-only block (e.g. UNKNOWN-DIAG-07A) that consumes the derived label inside the operator routing review queue ONLY, with no auto-accept, no clinical interpretation, no value parsing, and review-bound status preserved. The deferred pools (candidate_metadata_propagation_audit_pool, candidate_latin_medical_abbreviation_handling_audit_pool, candidate_table_header_language_policy record, likely_text_layer_issue, fallback_ran_but_no_family_match, and ambiguous_below_threshold) remain deferred or excluded; cue expansion remains not recommended for any subset.

## What this implementation does not change

- OCR routing logic
- OCR engine
- Classifier behavior for records OUTSIDE the exact 14-field signature
- Confidence thresholds or scoring
- Cue packs
- Auto-accept or review-bound policy
- B07 terminology, ROUTE-FIX, DB schema, command allowlist, external APIs

## Helper contract

The public helper `derive_numeric_table_safe_default_label(record, *, enabled=False)` is pure. It does not mutate the record. It returns the derived label only when the explicit `enabled=True` flag is passed AND the record matches every field of the 14-field positive signature AND no exclusion rule fires AND no implementation-level safeguard fires. Otherwise it returns `None`.
