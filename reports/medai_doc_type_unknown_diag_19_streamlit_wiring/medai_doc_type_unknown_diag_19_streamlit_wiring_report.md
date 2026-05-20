# MEDAI-DOC-TYPE-UNKNOWN-DIAG-19 Env-Gated Streamlit Wiring for PDF Text/Layout Quality

Default-off Streamlit wiring of the DIAG-18 read-only render plan into the Run & Review tab's Advanced technical details expander. The wiring renders only when BOTH the DIAG-17 metadata env var AND the DIAG-18 UI env var are truthy.

## State

- Phase ID: `MEDAI-DOC-TYPE-UNKNOWN-DIAG-19`
- Mode: `default_off_streamlit_wiring`
- Default off: **True**
- Metadata env var: `MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_IMPL_ENABLED`
- UI env var: `MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_UI_ENABLED`
- Requires both env vars truthy: **True**
- Branch: `clinical-knowledge-architecture`
- HEAD commit (short): `8eee257`
- DIAG-17 commit (short): `ad7b2d6`
- DIAG-18 commit (short): `f3c9760`
- PARK-20 parking commit (short): `3e46461`
- PARK-21 parking commit (short): `9f9e22d`
- PARK-22 parking commit (short): `f4d3cc6`

## PARK status

PARK-20 tags remain on origin at 3e46461. PARK-21 tags remain on origin at 9f9e22d. PARK-22 tags remain on origin at f4d3cc6. DIAG-19 does not touch any tag.

## Source reports referenced

- `block DIAG-13A (directory: medai_doc_type_unknown_diag_13a_text_layer_diagnostic)`
- `block DIAG-14 (directory: medai_doc_type_unknown_diag_14_text_layer_extraction_spec)`
- `block DIAG-15 (directory: medai_doc_type_unknown_diag_15_pdf_text_extraction_quality_audit)`
- `block DIAG-15B (directory: medai_doc_type_unknown_diag_15b_layout_table_extraction_audit)`
- `block DIAG-16 (directory: medai_doc_type_unknown_diag_16_pdf_text_layout_quality_spec)`
- `block DIAG-17 (directory: medai_doc_type_unknown_diag_17_pdf_text_layout_quality_impl)`
- `block DIAG-17B (directory: medai_doc_type_unknown_diag_17b_env_on_aggregate_eval)`
- `block DIAG-18 (directory: medai_doc_type_unknown_diag_18_pdf_text_layout_quality_ui)`

## Static audit of `app/main.py`

- `occurrence_count`: 1
- `block_line_count`: 27
- `inside_advanced_technical_details_expander`: True
- `after_prior_diag_08a_10a_12a_blocks`: True
- `has_try_except_guard`: True
- `has_diag_18_import_inside_try`: True
- `has_render_plan_call_inside_try`: True
- `forbidden_st_symbols_present`: []
- `forbidden_tokens_present`: []
- `unsafe_st_symbols_used_in_block`: []
- Safe `st.*` symbols used in block: ['st.caption', 'st.markdown']

## Env-combination audit (DIAG-18 helper)

- `neither_env_plan_is_none`: True
- `metadata_only_env_plan_is_none`: True
- `ui_only_env_plan_is_none`: True
- `both_env_plan_is_dict`: True
- `helpers_still_default_disabled_after_audit`: True

## Default-off proof

- `neither_env_plan_is_none`: True
- `metadata_only_env_plan_is_none`: True
- `ui_only_env_plan_is_none`: True
- `both_env_plan_is_dict`: True
- `helpers_still_default_disabled_after_audit`: True

## Read-only proof

- `diag_19_wiring_appears_exactly_once`: True
- `inside_advanced_technical_details_expander`: True
- `after_prior_diag_08a_10a_12a_blocks`: True
- `has_try_except_guard`: True
- `has_diag_18_import_inside_try`: True
- `has_render_plan_call_inside_try`: True
- `no_forbidden_st_symbols_in_block`: True
- `no_forbidden_tokens_in_block`: True
- `no_unsafe_st_symbols_used_in_block`: True

## Block invariants

- `behavior_changed`: True
- `default_behavior_changed`: False
- `runtime_behavior_changed_by_default`: False
- `streamlit_wiring_added`: True
- `streamlit_wiring_enabled_by_default`: False
- `advanced_technical_details_only`: True
- `buttons_added`: False
- `callbacks_added`: False
- `actions_added`: False
- `forms_added`: False
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
- `park_22_tags_touched`: False
- `all_records_review_bound`: True
- `accepted_count`: 0
- `auto_accept_allowed_count`: 0
- `external_api_used_count`: 0

## Safety / privacy

DIAG-19 adds a strictly default-off Streamlit wiring block in app/main.py::render_run_result_card's Advanced technical details expander. The block renders only when BOTH MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_IMPL_ENABLED and MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_UI_ENABLED are truthy. The DIAG-18 helper enforces the two-env-var gate internally; if either env var is unset or falsy the helper returns None and the block renders nothing. The block uses only st.markdown and st.caption — no buttons, no forms, no callbacks, no actions, no state mutation, no data-layer writes, no document_type mutation. The block carries no raw text, raw filenames, or private paths. The DIAG-18 helper import lives inside a try/except so non-Streamlit test collection is unaffected if either helper module is absent. No clinical interpretation; no diagnosis / medication / DDI / treatment inference; no abbreviation parsing or expansion; no external API enablement. PARK-20 / PARK-21 / PARK-22 tags are not touched.

## Progress estimate

| Track | Before | After |
| --- | --- | --- |
| Residual Unknown-reduction | ~99.98% done / ~0.02% remaining | ~99.99% done / ~0.01% remaining |
| Whole MedAI project | ~91% done / ~9% remaining | ~91.5% done / ~8.5% remaining |

## Recommended next block

- PARK-23 — parking snapshot capturing the DIAG-19 wiring (reports + tags only). After PARK-23, the natural follow-on is either a corpus-side env-on operator UAT block (still default-off in production) or an evaluation-only block that audits the wiring under Streamlit fixture tests. Cue expansion remains explicitly NOT recommended.

