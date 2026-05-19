# MEDAI-DOC-TYPE-UNKNOWN-DIAG-18 Read-Only PDF Text/Layout Quality Operator Surface

Default-off, read-only operator-surface helper for DIAG-17 metadata. Pure data-only render plan; no Streamlit widgets, no buttons, no callbacks, no actions, no state mutation.

## State

- Phase ID: `MEDAI-DOC-TYPE-UNKNOWN-DIAG-18`
- Mode: `default_off_read_only_operator_surface`
- Default off: **True**
- Read only: **True**
- Metadata env var: `MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_IMPL_ENABLED`
- UI env var: `MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_UI_ENABLED`
- Requires both env vars truthy: **True**
- Branch: `clinical-knowledge-architecture`
- HEAD commit (short): `212f73c`
- DIAG-17 commit (short): `ad7b2d6`
- DIAG-17B commit (short): `3bc8a64`
- PARK-20 parking commit (short): `3e46461`
- PARK-21 parking commit (short): `9f9e22d`

## PARK status

PARK-20 tags remain on origin at 3e46461. PARK-21 tags remain on origin at 9f9e22d. DIAG-18 does not touch any tag.

## Source reports referenced

- `block DIAG-13A (directory: medai_doc_type_unknown_diag_13a_text_layer_diagnostic)`
- `block DIAG-14 (directory: medai_doc_type_unknown_diag_14_text_layer_extraction_spec)`
- `block DIAG-15 (directory: medai_doc_type_unknown_diag_15_pdf_text_extraction_quality_audit)`
- `block DIAG-15B (directory: medai_doc_type_unknown_diag_15b_layout_table_extraction_audit)`
- `block DIAG-16 (directory: medai_doc_type_unknown_diag_16_pdf_text_layout_quality_spec)`
- `block DIAG-17 (directory: medai_doc_type_unknown_diag_17_pdf_text_layout_quality_impl)`
- `block DIAG-17B (directory: medai_doc_type_unknown_diag_17b_env_on_aggregate_eval)`

## Env-combination audit

| Env combination | In-scope emissions (of 21) | Excluded-pool emissions |
| --- | ---: | ---: |
| Neither | 0 | 0 of 3 |
| Metadata env truthy only | 0 | 0 of 3 |
| UI env truthy only | 0 | 0 of 3 |
| Both truthy | 21 | 0 of 3 |

## Default-off proof

- `neither_env_in_scope_emissions_must_be_zero`: True
- `metadata_only_env_in_scope_emissions_must_be_zero`: True
- `ui_only_env_in_scope_emissions_must_be_zero`: True
- `both_env_in_scope_emissions_must_equal_21`: True
- `both_env_excluded_pool_emissions_must_be_zero`: True
- `helper_still_default_disabled_after_audit`: True

## Read-only proof

- `forbidden_plan_keys_present`: []
- `is_read_only_count_must_equal_21`: True
- `no_action_attached_count_must_equal_21`: True
- `no_button_attached_count_must_equal_21`: True
- `no_callback_attached_count_must_equal_21`: True
- `no_state_mutation_count_must_equal_21`: True
- `no_data_layer_write_count_must_equal_21`: True
- `no_document_type_mutation_count_must_equal_21`: True
- `raw_text_rendered_count_must_be_zero`: True
- `raw_filename_rendered_count_must_be_zero`: True
- `private_path_rendered_count_must_be_zero`: True
- `clinical_interpretation_count_must_be_zero`: True
- `auto_accept_count_must_be_zero`: True
- `diagnosis_medication_ddi_treatment_inference_counts_must_be_zero`: True
- `abbreviation_expanded_count_must_be_zero`: True
- `external_api_used_count_must_be_zero`: True
- `park_20_touch_count_must_be_zero`: True
- `park_21_touch_count_must_be_zero`: True

## Block invariants

- `behavior_changed`: True
- `default_behavior_changed`: False
- `runtime_behavior_changed_by_default`: False
- `operator_ui_surface_added`: True
- `operator_ui_surface_enabled_by_default`: False
- `buttons_added`: False
- `callbacks_added`: False
- `actions_added`: False
- `state_mutation_added`: False
- `data_layer_write_added`: False
- `document_type_mutation_added`: False
- `external_api_used`: False
- `source_documents_opened`: False
- `raw_text_printed`: False
- `raw_filenames_printed`: False
- `private_paths_printed`: False
- `raw_text_rendered`: False
- `raw_filenames_rendered`: False
- `private_paths_rendered`: False
- `clinical_value_parsing_performed`: False
- `diagnosis_inference_performed`: False
- `medication_inference_performed`: False
- `ddi_inference_performed`: False
- `treatment_inference_performed`: False
- `abbreviation_expansion_performed`: False
- `cue_expansion_recommended`: False
- `cue_expansion_performed`: False
- `park_20_tags_touched`: False
- `park_21_tags_touched`: False
- `all_records_review_bound`: True
- `accepted_count`: 0
- `auto_accept_allowed_count`: 0
- `external_api_used_count`: 0

## Safety / privacy

DIAG-18 introduces a strictly default-off, read-only operator surface for the DIAG-17 PDF text/layout quality metadata. The surface renders only when BOTH MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_IMPL_ENABLED and MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_UI_ENABLED are truthy. If either is unset or falsy, no plan is produced, no metadata is generated, no side effect occurs. The render plan is pure data: no Streamlit widgets, no buttons, no callbacks, no actions, no accept/reject semantics, no state mutation, no data-layer writes, no document_type mutation. Plans use only controlled-vocabulary tokens; they never carry raw text, raw OCR text, raw document text, raw filenames, private paths, PHI, or secrets. No clinical interpretation; no diagnosis / medication / DDI / treatment inference; no abbreviation parsing or expansion; no external API enablement. PARK-20 / PARK-21 tags are not touched.

## Progress estimate

| Track | Before | After |
| --- | --- | --- |
| Residual Unknown-reduction | ~99.97% done / ~0.03% remaining | ~99.98% done / ~0.02% remaining |
| Whole MedAI project | ~90.8% done / ~9.2% remaining | ~91% done / ~9% remaining |

## Recommended next block

- PARK-22 — parking snapshot capturing the DIAG-17 + DIAG-17B + DIAG-18 trio (default-off helper, env-on aggregate evaluation, env-gated read-only operator surface). Reports + tags only; no runtime wiring. Cue expansion remains explicitly NOT recommended.

