# MEDAI-DOC-TYPE-UNKNOWN-DIAG-08A - Read-only Operator Badge UI

- branch: `clinical-knowledge-architecture`
- HEAD commit (short): `c78915e39244`
- source DIAG-07A commit (short): `8d8895d`
- PARK-19 baseline commit (short): `ac466e0f9ab8`
- public_report_commit_hash_policy: `short_hashes_only`
- upstream:
  - `reports/medai_doc_type_unknown_diag_07a_operator_routing_review/(public)`
  - `reports/medai_doc_type_unknown_diag_06a_implementation/(public)`
  - `reports/medai_doc_type_unknown_diag_06a/(public spec)`
  - `reports/medai_doc_type_unknown_diag_05/(public diagnostic)`
  - `reports/medai_doc_type_unknown_diag_04/(public diagnostic)`
  - `reports/medai_doc_type_unknown_diag_03/(public diagnostic)`
  - `reports/medai_doc_type_unknown_diag_02/(public diagnostic)`
  - `reports/medai_doc_type_unknown_diag_01/(public diagnostic)`
- source report: `reports/medai_doc_type_family_04_larger_slice_validation/(public anonymized batch-eval per-file table)`
- ui_surface_touched: `app/main.py::render_run_result_card -> `Advanced technical details` expander, read-only optional badge block`
- env_flag: `MEDAI_DOC_TYPE_OPERATOR_REVIEW_BADGE_ENABLED`
- expander label: `Operator review metadata`
- disclaimer line: `Review metadata only. Not a final document type. Not clinical interpretation.`
- generated_at: `2026-05-19T03:38:36.731853+00:00`

## UI integration summary

Adds a single read-only optional render block inside the existing `Advanced technical details` expander in `app/main.py::render_run_result_card`. The block calls the pure `render_plan_for_operator_badge` helper (`clinical_knowledge.document_type.operator_badge_ui`) and renders the returned plan via `st.markdown` / `st.caption` only. The block is wrapped in a defensive try/except so any import or render error is silently swallowed and the main result card is never blocked. Default-off: when the env var `MEDAI_DOC_TYPE_OPERATOR_REVIEW_BADGE_ENABLED` is unset / falsy, the helper returns None and the UI surface is visually unchanged.

## Disabled-state behavior

When the env var is unset or set to a falsy value, the helper returns None, the `if _op_badge_plan is not None` guard evaluates False, and no badge markdown is emitted. The expander content is identical to the pre-DIAG-08A state. Number of plans rendered on the 507-file corpus in this mode: 0.

## Enabled-state behavior

When the env var is set to a truthy value AND the record matches the exact 14-field DIAG-06A signature without violating any exclusion rule or implementation safeguard, the helper returns a structured render plan and the UI renders three markdown lines plus a disclaimer caption inside the existing expander. The badge is read-only; no button, form, or callback is attached. Records are not mutated; review-bound status is preserved. Number of plans rendered on the 507-file corpus in this mode: 11.

## Flag / rollback path

Four independent rollback paths, any one of which is sufficient: (1) omit / unset the env var `MEDAI_DOC_TYPE_OPERATOR_REVIEW_BADGE_ENABLED` (the default); (2) set the env var to a falsy value (`0` / `false` / `no` / `off` / `disabled`); (3) call the helper with `enabled=False`; (4) never import the helper - the existing main.py call site is wrapped in try/except so an ImportError is silently swallowed.

## Mode audits (corpus-wide)

### default_off
- plan_count: `0`
- matches_priority_slice_exactly: `False`
- extras_outside_priority_count: `0`
- missing_from_priority_count: `11`

### env_enabled
- plan_count: `11`
- matches_priority_slice_exactly: `True`
- extras_outside_priority_count: `0`
- missing_from_priority_count: `0`

### explicit_enabled
- plan_count: `11`
- matches_priority_slice_exactly: `True`
- extras_outside_priority_count: `0`
- missing_from_priority_count: `0`

## 11-record replay

- priority_slice_size: `11`
- enabled_true_plan_count: `11`
- enabled_false_plan_count: `0`
- default_off_plan_count: `0`
- matches_priority_slice_exactly: `True`

## 507-file aggregate

- corpus_size: `507`
- default_off_plan_count: `0`
- env_enabled_plan_count: `11`
- explicit_enabled_plan_count: `11`
- no_false_positive_outside_priority: `True`
- no_false_negative_inside_priority: `True`

## Counts

- operator_badge_display_count (with helper enabled): `11`
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

## No-action-attached confirmation

- no_action_attached_to_badge: `True`

## False-positive audit

- treatment_or_schedule_expansion: `0`
- imaging_expansion: `0`
- administrative_or_table_expansion: `0`
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

- before_08a_unknown_track_done_pct: `approximately 72%`
- before_08a_unknown_track_remaining_pct: `approximately 28%`
- before_08a_project_done_pct: `approximately 78%`
- before_08a_project_remaining_pct: `approximately 22%`
- after_08a_unknown_track_done_pct: `approximately 76%`
- after_08a_unknown_track_remaining_pct: `approximately 24%`
- after_08a_project_done_pct: `approximately 79%`
- after_08a_project_remaining_pct: `approximately 21%`
- note: `Estimates are approximate and refer to the residual Unknown-reduction track in this workspace, plus the overall MedAI project state. They are informational only and not a release milestone.`

## Safety / Privacy

- behavior_changed: `True` (strictly limited to read-only UI display)
- behavior_change_scope: Strictly limited to a single optional read-only badge render block inside the existing `Advanced technical details` expander in the Run & Review result card. No buttons, forms, or callbacks are attached. No clinical interpretation, no value parsing, no auto-accept, no active clinical fact writes, no document-type promotion at the data layer. The block is gated by the existing DIAG-07A env var and is OFF by default.
- clinical_behavior_changed: `False`
- external_api_used: `False`
- cue_expansion_recommended: `False`
- behavior_changed_strictly_limited_to_read_only_ui_display: `True`
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
- operator_badge_ui_default_disabled: `True`
- rollback_path_present: `True`
- no_action_attached_to_badge: `True`
- no_button_or_callback_in_render_plan: `True`
- ui_render_failure_is_silently_swallowed: `True`

No raw filenames, raw OCR text, raw document text, private paths, PHI, or secrets are included. The runtime behavior change is strictly limited to a single optional read-only badge block inside the existing Advanced technical details expander. No buttons, forms, or callbacks attach to the badge. No clinical interpretation, no value parsing, no auto-accept, no active clinical fact writes, no document-type promotion. Review-bound status preserved.

## Recommendation for next block

The read-only operator badge is now wired into the existing Advanced technical details expander, gated by the same env var and OFF by default. A future evaluation-only block (e.g. UNKNOWN-DIAG-09A) may consume operator-side click counts in an anonymized aggregate to assess whether the badge improves the operator review queue throughput; that block must remain review-bound, must not add auto-accept, and must not modify any classifier behavior. The deferred pools (1 table-header record, 11 propagation-audit, 8 abbreviation, 21 text-layer, 17 fallback, 15 ambiguous) remain deferred or excluded; cue expansion remains not recommended.

## What this block did not change

- OCR routing logic
- OCR engine
- Raw language / script detector behavior
- Classifier behavior for any record outside the exact 14-field signature
- Confidence thresholds or scoring
- Cue packs
- Auto-accept or review-bound rules
- Document-type promotion at the data layer
- B07 terminology, ROUTE-FIX, DB schema, command allowlist, external APIs
