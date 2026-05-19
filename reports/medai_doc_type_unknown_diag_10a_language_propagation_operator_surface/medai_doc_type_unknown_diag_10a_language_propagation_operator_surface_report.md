# MEDAI-DOC-TYPE-UNKNOWN-DIAG-10A - Language Propagation Operator Surface

- branch: `clinical-knowledge-architecture`
- HEAD commit (short): `40691604a4cb`
- source DIAG-09A-IMPLEMENTATION commit (short): `31f42fc`
- PARK-19 baseline commit (short): `ac466e0f9ab8`
- public_report_commit_hash_policy: `short_hashes_only`
- upstream:
  - `reports/medai_doc_type_unknown_diag_09a_implementation/(public)`
  - `reports/medai_doc_type_unknown_diag_09a/(public spec)`
  - `reports/medai_doc_type_unknown_diag_08a_operator_badge_ui/(public)`
  - `reports/medai_doc_type_unknown_diag_07a_operator_routing_review/(public)`
  - `reports/medai_doc_type_unknown_diag_06a_implementation/(public)`
  - `reports/medai_doc_type_unknown_diag_06a/(public spec)`
- source report: `reports/medai_doc_type_family_04_larger_slice_validation/(public anonymized batch-eval per-file table)`
- ui_surface_touched: `app/main.py::render_run_result_card -> `Advanced technical details` expander, optional read-only language-propagation metadata block (rendered alongside but distinct from the DIAG-08A operator badge block)`
- propagation_env_flag: `MEDAI_DOC_TYPE_LANGUAGE_PROPAGATION_METADATA_ENABLED`
- operator_review_env_flag (distinct): `MEDAI_DOC_TYPE_OPERATOR_REVIEW_BADGE_ENABLED`
- propagation_display_text: `metadata: Latin detector likely-English context - review required`
- propagation_vocab_token: `latin_detector_likely_english_context_review_required`
- propagation_disclaimer_line: `Review metadata only. Not a final document type. Not clinical interpretation.`
- propagation_expander_label: `Language propagation metadata`
- generated_at: `2026-05-19T09:24:37.341686+00:00`

## Operator-surface integration summary

Adds `clinical_knowledge.document_type.render_plan_for_language_propagation` as a pure data-only render-plan helper that consumes the DIAG-09A-IMPLEMENTATION propagation helper and returns a structured plan with `expander_label`, three `markdown_lines` (propagation display text, vocab token, source label), a `disclaimer_line` ('Review metadata only. Not a final document type. Not clinical interpretation.'), and explicit `is_read_only` / `no_action_attached` / `review_bound` / `is_clinical_classification=False` / `is_final_document_type=False` / `is_auto_accept=False` / `raw_detector_output_unchanged=True` / `is_data_layer_document_type_change=False` flags. A small optional render block in `app/main.py::render_run_result_card` (inside the existing `Advanced technical details` expander, rendered alongside but distinct from the DIAG-08A operator-badge block) lazily imports the helper, calls it, and emits the badge text and disclaimer via `st.markdown` and `st.caption` only. The block is wrapped in `try/except Exception: pass`. Default-off; rendered only when the SEPARATE env var `MEDAI_DOC_TYPE_LANGUAGE_PROPAGATION_METADATA_ENABLED` is truthy. The DIAG-07A env var (`MEDAI_DOC_TYPE_OPERATOR_REVIEW_BADGE_ENABLED`) does NOT enable this display.

## Disabled-state behavior

When the propagation env var is unset or set to a falsy value, the helper returns None, the `if _lp_plan is not None` guard evaluates False, and no markdown is emitted for this lever. The expander content reflects only the DIAG-08A operator badge (if its own env var is set) or nothing at all. Number of propagation plans rendered on the 507-file corpus in this mode: 0.

## Enabled-state behavior

When the propagation env var is set to a truthy value AND the record matches the exact 11-field propagation signature without violating any exclusion rule, implementation safeguard, or numeric-table overlap check, the helper returns a structured render plan and the UI emits three markdown lines plus a disclaimer caption inside the existing expander. The display is read-only; no button, form, or callback is attached. Records are not mutated; review-bound status is preserved; raw detector output is unchanged; data-layer document type is unchanged. Number of propagation plans rendered on the 507-file corpus in this mode: 11.

## Flag / rollback path

Multiple rollback paths exist, any one of which is sufficient: (1) leave the env var `MEDAI_DOC_TYPE_LANGUAGE_PROPAGATION_METADATA_ENABLED` unset; (2) set it to a falsy value; (3) pass `enabled=False` to the helper explicitly; (4) never import the operator-surface module. No persisted state to roll back. The function is pure. The DIAG-08A operator-badge env var is independently togglable and toggling it has no effect on this lever.

## Flag-separation audit

- default_off_propagation_plans_zero: `True`
- default_off_operator_badge_plans_zero: `True`
- prop_env_only_yields_propagation_plans: `True`
- prop_env_only_yields_zero_operator_badge_plans: `True`
- op_env_only_yields_zero_propagation_plans: `True`
- op_env_only_yields_operator_badge_plans: `True`
- both_env_yields_both_levers_with_correct_priority_slices: `True`
- flag_separation_holds_in_all_modes: `True`

## Mode audits (corpus-wide)

### default_off
- propagation_plan_count: `0`
- operator_badge_plan_count: `0`
- propagation_matches_priority_slice_exactly: `False`
- operator_badge_matches_priority_slice_exactly: `False`
- propagation_extras_outside_priority_count: `0`
- operator_badge_extras_outside_priority_count: `0`

### propagation_env_only
- propagation_plan_count: `11`
- operator_badge_plan_count: `0`
- propagation_matches_priority_slice_exactly: `True`
- operator_badge_matches_priority_slice_exactly: `False`
- propagation_extras_outside_priority_count: `0`
- operator_badge_extras_outside_priority_count: `0`

### operator_badge_env_only
- propagation_plan_count: `0`
- operator_badge_plan_count: `11`
- propagation_matches_priority_slice_exactly: `False`
- operator_badge_matches_priority_slice_exactly: `True`
- propagation_extras_outside_priority_count: `0`
- operator_badge_extras_outside_priority_count: `0`

### both_env
- propagation_plan_count: `11`
- operator_badge_plan_count: `11`
- propagation_matches_priority_slice_exactly: `True`
- operator_badge_matches_priority_slice_exactly: `True`
- propagation_extras_outside_priority_count: `0`
- operator_badge_extras_outside_priority_count: `0`

## 11-record propagation replay

- priority_slice_size: `11`
- enabled_true_plan_count: `11`
- enabled_false_plan_count: `0`
- default_off_plan_count: `0`
- matches_priority_slice_exactly: `True`

## 507-file aggregate

- corpus_size: `507`
- default_off_propagation_plan_count: `0`
- default_off_operator_badge_plan_count: `0`
- prop_env_only_propagation_plan_count: `11`
- prop_env_only_operator_badge_plan_count: `0`
- op_env_only_propagation_plan_count: `0`
- op_env_only_operator_badge_plan_count: `11`
- both_env_propagation_plan_count: `11`
- both_env_operator_badge_plan_count: `11`
- no_false_positive_outside_priority: `True`
- no_false_negative_inside_priority: `True`

## Counts

- propagation_metadata_display_count (propagation env enabled): `11`
- numeric_table_badge_display_count (operator-badge env enabled): `11`
- unknown_count_at_data_layer_before: `107`
- unknown_count_at_data_layer_after: `107`
- unknown_count_at_data_layer_delta: `0`
- accepted_count: `0`
- auto_accept_allowed_count: `0`
- external_api_used_count: `0`

## Review-bound preservation

- review_bound_records_before: `507`
- review_bound_records_after: `507`
- review_bound_preserved: `True`

## No-action / no-mutation confirmation

- no_action_attached_to_plan: `True`
- raw_detector_output_unchanged: `True`
- data_layer_document_type_unchanged: `True`

## False-positive audit

- numeric_table_overlap: `0`
- treatment_or_schedule_expansion: `0`
- imaging_expansion: `0`
- administrative_or_table_expansion: `0`
- other_expansion: `0`
- no_false_positive_expansion: `True`

## Deferred subsets (out of scope)

- numeric_table_safe_default_pool_handled_by_diag06_07_08: 11 records covered by DIAG-06A/07A/08A; separate badge lever
- candidate_latin_medical_abbreviation_handling_audit_pool: 8 records from DIAG-04 routed to the abbreviation lever; deferred
- candidate_table_header_language_policy_record: 1 record from DIAG-05 routed to the table-header lever; deferred
- likely_text_layer_issue: 21 records deferred per DIAG-03
- fallback_ran_but_no_family_match: 17 records deferred per DIAG-02; no cue expansion
- ambiguous_below_threshold: 15 records excluded; review-bound, no cue expansion

## Progress estimate

- before_10a_unknown_track_done_pct: `approximately 84%`
- before_10a_unknown_track_remaining_pct: `approximately 16%`
- before_10a_project_done_pct: `approximately 81%`
- before_10a_project_remaining_pct: `approximately 19%`
- after_10a_unknown_track_done_pct: `approximately 87%`
- after_10a_unknown_track_remaining_pct: `approximately 13%`
- after_10a_project_done_pct: `approximately 82%`
- after_10a_project_remaining_pct: `approximately 18%`
- note: `Estimates are approximate and refer to the residual Unknown-reduction track in this workspace, plus the overall MedAI project state. They are informational only and not a release milestone.`

## Safety / Privacy

- behavior_changed: `True` (strictly limited to read-only UI display)
- behavior_change_scope: Strictly limited to a single optional read-only language-propagation metadata block inside the existing `Advanced technical details` expander in the Run & Review result card. Gated by the SEPARATE propagation env var; the DIAG-07A operator-badge env var does not enable this lever. No buttons, forms, or callbacks attached. No clinical interpretation, no value parsing, no auto-accept, no active clinical fact writes, no document-type promotion at the data layer, no raw-detector-output mutation.
- clinical_behavior_changed: `False`
- external_api_used: `False`
- cue_expansion_recommended: `False`
- behavior_changed_strictly_limited_to_read_only_ui_display: `True`
- clinical_behavior_changed: `False`
- ocr_routing_changed: `False`
- ocr_engine_changed: `False`
- raw_language_detector_behavior_changed: `False`
- raw_detector_output_unchanged: `True`
- data_layer_document_type_unchanged: `True`
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
- operator_surface_default_disabled: `True`
- rollback_path_present: `True`
- no_action_attached_to_plan: `True`
- no_button_or_callback_in_render_plan: `True`
- ui_render_failure_is_silently_swallowed: `True`
- flag_separation_holds: `True`

No raw filenames, raw OCR text, raw document text, private paths, PHI, or secrets are included. Read-only operator display metadata only. No clinical interpretation, no value parsing, no auto-accept, no active clinical fact writes, no document-type promotion, no raw-detector-output mutation. Review-bound status preserved. Flag separation from the DIAG-07A operator-badge env var is preserved in every mode.
