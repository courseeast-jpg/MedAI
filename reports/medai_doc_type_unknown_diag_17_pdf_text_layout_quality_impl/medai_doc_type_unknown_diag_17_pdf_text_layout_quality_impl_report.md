# MEDAI-DOC-TYPE-UNKNOWN-DIAG-17 Default-Off PDF Text/Layout Quality Implementation Pass

First default-off, env-gated implementation pass under the DIAG-16 acceptance criteria. Pure helper; not wired into any runtime path. Default-off proof recorded below.

## State

- Phase ID: `MEDAI-DOC-TYPE-UNKNOWN-DIAG-17`
- Mode: `default_off_implementation`
- Default off: **True**
- Env var: `MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_IMPL_ENABLED`
- Branch: `clinical-knowledge-architecture`
- HEAD commit (short): `b9db9d5`
- DIAG-16 commit (short): `e144376`
- PARK-20 parking commit (short): `3e46461`
- PARK-21 parking commit (short): `9f9e22d`

## PARK status

PARK-20 tags remain on origin at 3e46461. PARK-21 tags remain on origin at 9f9e22d. DIAG-17 does not touch any tag.

## Source reports referenced

- `block DIAG-13A (directory: medai_doc_type_unknown_diag_13a_text_layer_diagnostic)`
- `block DIAG-14 (directory: medai_doc_type_unknown_diag_14_text_layer_extraction_spec)`
- `block DIAG-15 (directory: medai_doc_type_unknown_diag_15_pdf_text_extraction_quality_audit)`
- `block DIAG-15B (directory: medai_doc_type_unknown_diag_15b_layout_table_extraction_audit)`
- `block DIAG-16 (directory: medai_doc_type_unknown_diag_16_pdf_text_layout_quality_spec)`

## Total records in scope: **21** (11 Sub-track A + 10 Sub-track B)

## Env-mode audit

| Mode | Target emissions (of 21) | Excluded-pool emissions |
| --- | ---: | ---: |
| Default off (env empty) | 0 of 21 | 0 of 4 |
| Explicit off (env truthy, kwarg False) | 0 of 21 | 0 of 4 |
| Explicit on (env truthy) | 21 of 21 | 0 of 4 |

## Family-label counts (multi-label) under explicit-on

| Quality family label | Count |
| --- | ---: |
| `pdf_text_too_short` | 11 |
| `table_structure_visible_text_insufficient` | 10 |
| `layout_or_table_extraction_gap` | 10 |

## Default-off proof

- `default_off_emissions_must_be_zero`: True
- `explicit_off_emissions_must_be_zero`: True
- `explicit_on_target_emissions_must_equal_21`: True
- `explicit_on_excluded_emissions_must_be_zero`: True
- `review_required_all_true`: True
- `auto_accept_allowed_any_true`: False
- `clinical_interpretation_any_true`: False
- `raw_text_any_emitted`: False
- `raw_filename_any_emitted`: False
- `private_path_any_emitted`: False

## Block invariants

- `behavior_changed`: True
- `default_behavior_changed`: False
- `external_api_used`: False
- `source_documents_opened`: False
- `raw_text_printed`: False
- `raw_filenames_printed`: False
- `private_paths_printed`: False
- `runtime_behavior_changed_by_default`: False
- `extraction_behavior_changed_by_default`: False
- `pdf_text_extraction_behavior_changed_by_default`: False
- `layout_extraction_behavior_changed_by_default`: False
- `table_extraction_behavior_changed_by_default`: False
- `ocr_behavior_changed`: False
- `classifier_behavior_changed`: False
- `threshold_behavior_changed`: False
- `cue_expansion_recommended`: False
- `cue_expansion_performed`: False
- `operator_ui_surface_added`: False
- `clinical_value_parsing_performed`: False
- `diagnosis_inference_performed`: False
- `medication_inference_performed`: False
- `ddi_inference_performed`: False
- `treatment_inference_performed`: False
- `abbreviations_parsed_or_expanded`: False
- `park_20_tags_touched`: False
- `park_21_tags_touched`: False
- `all_records_review_bound`: True
- `accepted_count`: 0
- `auto_accept_allowed_count`: 0
- `external_api_used_count`: 0

## Safety / privacy

DIAG-17 introduces a strictly default-off, env-gated PDF text/layout quality metadata helper. With no kwarg and the SEPARATE env var MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_IMPL_ENABLED unset (or falsy), the helper returns None for every input. When explicitly enabled, the helper emits only controlled-vocabulary metadata derived from privacy-safe DIAG-13A/14/15/15B aggregates: no raw extracted text, no raw OCR text, no raw document text, no raw filenames, no private paths, no PHI, no secrets. The helper performs no clinical interpretation, no diagnosis/medication/DDI/treatment inference, no clinical value parsing, no abbreviation parsing or expansion, no auto-accept, no data-layer document_type change, no OCR routing change, no OCR engine behavior change, no PDF text-extraction or layout/table extraction change, no classifier change, no threshold/scoring change, no cue expansion, no external API enablement. PARK-20 and PARK-21 tags are not touched.

## Progress estimate

| Track | Before | After |
| --- | --- | --- |
| Residual Unknown-reduction | ~99.9% done / ~0.1% remaining | ~99.95% done / ~0.05% remaining |
| Whole MedAI project | ~90% done / ~10% remaining | ~90.5% done / ~9.5% remaining |

## Recommended next block

- DIAG-18 — first env-gated wiring of the DIAG-17 helper into a narrow operator surface (read-only, no auto-accept) OR a separate evaluation block that captures aggregate metrics under the env-on path against the real privacy-safe corpus. Either next block must remain default-off, review-bound, and aggregate-only. Cue expansion remains explicitly NOT recommended.
- Must remain default off: **True**
- Must remain review-bound: **True**
- Must NOT propose cue expansion as primary step: **True**
- Must NOT change default behavior: **True**

