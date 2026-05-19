# MEDAI-DOC-TYPE-UNKNOWN-DIAG-07A - Operator Routing Review Integration

- branch: `clinical-knowledge-architecture`
- HEAD commit (short): `ed132836e44d`
- source implementation commit (short): `eef93bc`
- PARK-19 baseline commit (short): `ac466e0f9ab8`
- public_report_commit_hash_policy: `short_hashes_only`
- upstream:
  - `reports/medai_doc_type_unknown_diag_06a_implementation/(public)`
  - `reports/medai_doc_type_unknown_diag_06a/(public spec)`
  - `reports/medai_doc_type_unknown_diag_05/(public diagnostic)`
  - `reports/medai_doc_type_unknown_diag_04/(public diagnostic)`
  - `reports/medai_doc_type_unknown_diag_03/(public diagnostic)`
  - `reports/medai_doc_type_unknown_diag_02/(public diagnostic)`
  - `reports/medai_doc_type_unknown_diag_01/(public diagnostic)`
- source report: `reports/medai_doc_type_family_04_larger_slice_validation/(public anonymized batch-eval per-file table)`
- generated_at: `2026-05-19T03:18:10.010176+00:00`

- operator_review_badge_default_disabled: `True`
- operator_review_badge_env_var: `MEDAI_DOC_TYPE_OPERATOR_REVIEW_BADGE_ENABLED`
- operator_review_badge_vocab_token: `table_latin_likely_english_context_review_required`
- operator_review_badge_text: `metadata: table-Latin-likely-English-context - review required`
- operator_review_badge_disclaimer: `This is a routing-review hint for the operator queue. It is not a clinical classification, not a final document type, and does not auto-accept. The document remains review-bound.`

## Operator integration summary

Adds `clinical_knowledge.document_type.derive_operator_review_badge` as a thin, default-off consumer of the DIAG-06A helper. The badge is returned ONLY when the operator-review flag is explicitly enabled (via the `enabled=True` keyword argument OR the `MEDAI_DOC_TYPE_OPERATOR_REVIEW_BADGE_ENABLED` environment variable set to a truthy value) AND the underlying signature/exclusion guards from DIAG-06A all hold. The returned dict carries explicit `review_bound=True`, `is_clinical_classification=False`, `is_final_document_type=False`, `is_auto_accept=False`, and `is_active_clinical_fact=False` so consumers cannot mistake the badge for a clinical outcome. No mutation of the record, no auto-accept, no clinical interpretation, no value parsing, no active fact writes. The DIAG-06A helper remains default-off outside this explicit operator-review call site.

## Flag / rollback path

Multiple disable / rollback paths exist, any one of which is sufficient: (1) omit the `enabled` kwarg AND keep the env var `MEDAI_DOC_TYPE_OPERATOR_REVIEW_BADGE_ENABLED` unset (the default); (2) pass `enabled=False` explicitly; (3) unset the env var if it was set; (4) never import the integration module - existing pipelines are unaffected by the addition. No persisted state to roll back. The function is pure.

## Mode audits (corpus-wide)

### default_off
- badge_count: `0`
- matches_priority_slice_exactly: `False`
- extras_outside_priority_count: `0`
- missing_from_priority_count: `11`

### env_enabled
- badge_count: `11`
- matches_priority_slice_exactly: `True`
- extras_outside_priority_count: `0`
- missing_from_priority_count: `0`

### explicit_enabled
- badge_count: `11`
- matches_priority_slice_exactly: `True`
- extras_outside_priority_count: `0`
- missing_from_priority_count: `0`

## 11-record replay

- priority_slice_size: `11`
- enabled_true_badge_count: `11`
- enabled_false_badge_count: `0`
- default_off_badge_count: `0`
- matches_priority_slice_exactly: `True`

## 507-file aggregate

- corpus_size: `507`
- default_off_badge_count: `0`
- env_enabled_badge_count: `11`
- explicit_enabled_badge_count: `11`
- no_false_positive_outside_priority: `True`
- no_false_negative_inside_priority: `True`

## Counts

- operator_badge_display_count (with helper enabled): `11`
- unknown_count_at_data_layer_before: `107`
- unknown_count_at_data_layer_after: `107`
- unknown_count_at_data_layer_delta: `0`
- operator_review_metadata_display_count: `11` (separate from the data-layer unknown count)
- accepted_count: `0`
- auto_accept_allowed_count: `0`
- external_api_used_count: `0`

## Review-bound preservation

- review_bound_records_before: `507`
- review_bound_records_after: `507`
- review_bound_preserved: `True`

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

- before_07a_unknown_track_done_pct: `approximately 68%`
- before_07a_unknown_track_remaining_pct: `approximately 32%`
- before_07a_project_done_pct: `approximately 77%`
- before_07a_project_remaining_pct: `approximately 23%`
- after_07a_unknown_track_done_pct: `approximately 72%`
- after_07a_unknown_track_remaining_pct: `approximately 28%`
- after_07a_project_done_pct: `approximately 78%`
- after_07a_project_remaining_pct: `approximately 22%`
- note: `Estimates are approximate and refer to the residual Unknown-reduction track in this workspace, plus the overall MedAI project state. They are informational only and not a release milestone.`

## Safety / Privacy

- behavior_changed: `True` (strictly limited to the operator review display surface)
- behavior_change_scope: Strictly limited to deriving an operator-review display badge for records that match the exact 14-field positive signature AND only when the operator-review flag is explicitly enabled. No clinical interpretation, no value parsing, no auto-accept, no active clinical fact writes, no document-type promotion, no OCR routing change. The badge is read-only review metadata.
- clinical_behavior_changed: `False`
- external_api_used: `False`
- cue_expansion_recommended: `False`
- behavior_changed_strictly_limited_to_operator_review_display: `True`
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
- operator_review_badge_default_disabled: `True`
- rollback_path_present: `True`
- underlying_helper_default_disabled_outside_call_site: `True`

No raw filenames, raw OCR text, raw document text, private paths, PHI, or secrets are included. The runtime behavior change is strictly limited to deriving the operator-review badge for records that match the exact 14-field signature and only when the operator-review flag is explicitly enabled. No clinical interpretation, no value parsing, no auto-accept, no active clinical fact writes, no document-type promotion. Review-bound status is preserved.

## Recommendation for next block

The operator-review badge is available behind a default-off env-gated and kwarg-gated flag. A future evaluation-only block (e.g. UNKNOWN-DIAG-08A) may wire the badge into a small UI surface read-only display, preserving review-bound status and the existing operator review queue. Continue to leave the deferred pools (1 table-header record, 11 propagation-audit, 8 abbreviation, 21 text-layer, 17 fallback, 15 ambiguous) deferred or excluded; cue expansion remains not recommended.

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
