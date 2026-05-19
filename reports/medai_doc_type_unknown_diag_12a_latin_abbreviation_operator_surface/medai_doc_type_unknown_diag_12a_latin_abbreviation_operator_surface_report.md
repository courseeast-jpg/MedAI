# MEDAI-DOC-TYPE-UNKNOWN-DIAG-12A - Latin Abbreviation Operator Surface

- branch: `clinical-knowledge-architecture`
- HEAD commit (short): `bea4f2f63cca`
- source DIAG-11A-IMPLEMENTATION commit (short): `a51f323`
- PARK-19 baseline commit (short): `ac466e0f9ab8`
- public_report_commit_hash_policy: `short_hashes_only`
- upstream:
  - `reports/medai_doc_type_unknown_diag_11a_implementation/(public)`
  - `reports/medai_doc_type_unknown_diag_11a/(public spec)`
  - `reports/medai_doc_type_unknown_diag_10a_language_propagation_operator_surface/(public)`
  - `reports/medai_doc_type_unknown_diag_09a_implementation/(public)`
  - `reports/medai_doc_type_unknown_diag_08a_operator_badge_ui/(public)`
  - `reports/medai_doc_type_unknown_diag_07a_operator_routing_review/(public)`
  - `reports/medai_doc_type_unknown_diag_06a_implementation/(public)`
- source report: `reports/medai_doc_type_family_04_larger_slice_validation/(public anonymized batch-eval per-file table)`
- ui_surface_touched: `app/main.py::render_run_result_card -> `Advanced technical details` expander, third optional read-only block (Latin abbreviation metadata) rendered alongside but distinct from the DIAG-08A operator badge block and the DIAG-10A language-propagation block`
- abbreviation_env_flag: `MEDAI_DOC_TYPE_LATIN_ABBREVIATION_METADATA_ENABLED`
- propagation_env_flag (distinct): `MEDAI_DOC_TYPE_LANGUAGE_PROPAGATION_METADATA_ENABLED`
- operator_review_env_flag (distinct): `MEDAI_DOC_TYPE_OPERATOR_REVIEW_BADGE_ENABLED`
- abbreviation_display_text: `metadata: Latin medical abbreviation context - review required`
- abbreviation_vocab_token: `latin_medical_abbreviation_context_review_required`
- abbreviation_disclaimer_line: `Review metadata only. Not a final document type. Not clinical interpretation. Abbreviations are not parsed or expanded.`
- abbreviation_expander_label: `Latin abbreviation metadata`
- generated_at: `2026-05-19T10:04:15.759655+00:00`

## Operator-surface integration summary

Adds `clinical_knowledge.document_type.render_plan_for_latin_abbreviation` as a pure data-only render-plan helper that consumes the DIAG-11A-IMPLEMENTATION abbreviation helper and returns a structured plan with `expander_label`, three `markdown_lines` (abbreviation display text, vocab token, source label), a `disclaimer_line` ('Review metadata only. Not a final document type. Not clinical interpretation. Abbreviations are not parsed or expanded.'), and explicit `is_read_only` / `no_action_attached` / `review_bound` / `is_clinical_classification=False` / `is_final_document_type=False` / `is_auto_accept=False` / `is_data_layer_document_type_change=False` / `raw_detector_output_unchanged=True` / `abbreviation_parsed=False` / `abbreviation_expanded=False` flags. A third optional render block in `app/main.py::render_run_result_card` (inside the existing `Advanced technical details` expander, rendered alongside but distinct from the DIAG-08A operator-badge and DIAG-10A language-propagation blocks) lazily imports the helper and emits the markdown / caption via `st.markdown` / `st.caption` only. The block is wrapped in `try/except Exception: pass`. Default-off; rendered only when the SEPARATE env var `MEDAI_DOC_TYPE_LATIN_ABBREVIATION_METADATA_ENABLED` is truthy. Neither the `MEDAI_DOC_TYPE_OPERATOR_REVIEW_BADGE_ENABLED` env var nor the `MEDAI_DOC_TYPE_LANGUAGE_PROPAGATION_METADATA_ENABLED` env var enables this display - all three levers toggle independently.

## Disabled-state behavior

When the abbreviation env var is unset or set to a falsy value, the helper returns None, the `if _la_plan is not None` guard evaluates False, and no markdown is emitted for this lever. The expander reflects only whichever of the other two levers (DIAG-08A operator badge, DIAG-10A propagation) have their own env vars set, or nothing at all. Number of abbreviation plans on the 507-file corpus in this mode: 0.

## Enabled-state behavior

When the abbreviation env var is truthy AND the record matches the exact 14-field abbreviation signature without violating any exclusion rule, implementation safeguard, or overlap check with the numeric-table / propagation pools, the helper returns a structured render plan and the UI emits three markdown lines plus a disclaimer caption inside the existing expander. The display is read-only; the abbreviation is never parsed or expanded; no button, form, or callback is attached. Records are not mutated; review-bound preserved; raw detector output unchanged; data-layer document type unchanged. Number of abbreviation plans on the 507-file corpus in this mode: 8.

## Flag / rollback path

Four independent rollback paths, any one of which is sufficient: (1) leave the env var `MEDAI_DOC_TYPE_LATIN_ABBREVIATION_METADATA_ENABLED` unset; (2) set it to a falsy value; (3) pass `enabled=False` to the helper explicitly; (4) never import the operator-surface module. No persisted state to roll back. The function is pure. The DIAG-07A operator-badge env var and the DIAG-09A propagation env var are independently togglable and toggling either has no effect on this lever.

## Three-way flag-separation audit

- default_off_yields_zero_from_each_lever: `True`
- abbrev_env_only_yields_abbreviation_slice: `True`
- propagation_env_only_yields_propagation_slice_only: `True`
- operator_badge_env_only_yields_numeric_table_slice_only: `True`
- all_three_env_vars_yields_union_with_no_cross_contamination: `True`
- three_env_vars_are_distinct: `True`
- three_way_flag_separation_holds: `True`

## Mode matrix (corpus-wide)

### default_off
- abbreviation_plan_count: `0`
- propagation_plan_count: `0`
- operator_badge_plan_count: `0`

### abbrev_env_only
- abbreviation_plan_count: `8`
- propagation_plan_count: `0`
- operator_badge_plan_count: `0`

### propagation_env_only
- abbreviation_plan_count: `0`
- propagation_plan_count: `11`
- operator_badge_plan_count: `0`

### operator_badge_env_only
- abbreviation_plan_count: `0`
- propagation_plan_count: `0`
- operator_badge_plan_count: `11`

### abbrev_plus_propagation
- abbreviation_plan_count: `8`
- propagation_plan_count: `11`
- operator_badge_plan_count: `0`

### abbrev_plus_operator_badge
- abbreviation_plan_count: `8`
- propagation_plan_count: `0`
- operator_badge_plan_count: `11`

### propagation_plus_operator_badge
- abbreviation_plan_count: `0`
- propagation_plan_count: `11`
- operator_badge_plan_count: `11`

### all_three_env_vars
- abbreviation_plan_count: `8`
- propagation_plan_count: `11`
- operator_badge_plan_count: `11`

## 8-record abbreviation replay

- priority_slice_size: `8`
- enabled_true_plan_count: `8`
- enabled_false_plan_count: `0`
- default_off_plan_count: `0`
- matches_priority_slice_exactly: `True`

## 507-file aggregate

- corpus_size: `507`
- default_off_abbreviation_plan_count: `0`
- abbrev_env_only_abbreviation_plan_count: `8`
- propagation_env_only_abbreviation_plan_count: `0`
- operator_badge_env_only_abbreviation_plan_count: `0`
- abbrev_plus_propagation_abbreviation_plan_count: `8`
- abbrev_plus_operator_badge_abbreviation_plan_count: `8`
- propagation_plus_operator_badge_abbreviation_plan_count: `0`
- all_three_env_vars_abbreviation_plan_count: `8`
- default_off_propagation_plan_count: `0`
- abbrev_env_only_propagation_plan_count: `0`
- propagation_env_only_propagation_plan_count: `11`
- operator_badge_env_only_propagation_plan_count: `0`
- abbrev_plus_propagation_propagation_plan_count: `11`
- abbrev_plus_operator_badge_propagation_plan_count: `0`
- propagation_plus_operator_badge_propagation_plan_count: `11`
- all_three_env_vars_propagation_plan_count: `11`
- default_off_operator_badge_plan_count: `0`
- abbrev_env_only_operator_badge_plan_count: `0`
- propagation_env_only_operator_badge_plan_count: `0`
- operator_badge_env_only_operator_badge_plan_count: `11`
- abbrev_plus_propagation_operator_badge_plan_count: `0`
- abbrev_plus_operator_badge_operator_badge_plan_count: `11`
- propagation_plus_operator_badge_operator_badge_plan_count: `11`
- all_three_env_vars_operator_badge_plan_count: `11`
- no_false_positive_outside_priority: `True`
- no_false_negative_inside_priority: `True`

## Display counts (under each env var alone)

- abbreviation_metadata_display_count (abbreviation env enabled): `8`
- numeric_table_badge_display_count (operator-badge env enabled): `11`
- language_propagation_display_count (propagation env enabled): `11`

## Counts

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
- abbreviation_not_parsed: `True`
- abbreviation_not_expanded: `True`

## False-positive audit

- numeric_table_overlap: `0`
- language_propagation_overlap: `0`
- treatment_or_schedule_expansion: `0`
- imaging_expansion: `0`
- administrative_or_table_expansion: `0`
- other_expansion: `0`
- no_false_positive_expansion: `True`

## Deferred subsets (out of scope)

- numeric_table_safe_default_pool_handled_by_diag06_07_08: 11 records covered by DIAG-06A/07A/08A; separate badge lever
- language_propagation_pool_handled_by_diag09_10: 11 records covered by DIAG-09A/10A; separate propagation lever
- candidate_table_header_language_policy_record: 1 record from DIAG-05 routed to the table-header lever; deferred
- likely_text_layer_issue: 21 records deferred per DIAG-03
- fallback_ran_but_no_family_match: 17 records deferred per DIAG-02; no cue expansion
- ambiguous_below_threshold: 15 records excluded; review-bound, no cue expansion

## Progress estimate

- before_12a_unknown_track_done_pct: `approximately 93%`
- before_12a_unknown_track_remaining_pct: `approximately 7%`
- before_12a_project_done_pct: `approximately 84%`
- before_12a_project_remaining_pct: `approximately 16%`
- after_12a_unknown_track_done_pct: `approximately 96%`
- after_12a_unknown_track_remaining_pct: `approximately 4%`
- after_12a_project_done_pct: `approximately 85%`
- after_12a_project_remaining_pct: `approximately 15%`
- note: `Estimates are approximate and refer to the residual Unknown-reduction track in this workspace, plus the overall MedAI project state. They are informational only and not a release milestone.`

## Safety / Privacy

- behavior_changed: `True` (strictly limited to read-only UI display)
- behavior_change_scope: Strictly limited to a single optional read-only Latin abbreviation metadata block inside the existing `Advanced technical details` expander. Gated by the SEPARATE abbreviation env var; the DIAG-07A operator-badge env var and the DIAG-09A propagation env var never enable this lever. No buttons, forms, or callbacks attached. No abbreviation parsing or expansion. No clinical interpretation, no value parsing, no auto-accept, no active clinical fact writes, no document-type promotion, no raw detector output mutation.
- clinical_behavior_changed: `False`
- external_api_used: `False`
- cue_expansion_recommended: `False`
- abbreviation_parsing_or_expansion: `False`
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
- abbreviation_parsing_or_expansion_added: `False`
- abbreviation_parsed: `False`
- abbreviation_expanded: `False`
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
- three_way_flag_separation_holds: `True`

No raw filenames, raw OCR text, raw document text, private paths, PHI, or secrets are included. Read-only operator display metadata only. The abbreviation is never parsed or expanded. No clinical interpretation, no value parsing, no auto-accept, no active clinical fact writes, no document-type promotion, no raw-detector-output mutation. Review-bound status preserved. Three-way flag separation from the DIAG-07A and DIAG-09A levers is preserved in every mode.
