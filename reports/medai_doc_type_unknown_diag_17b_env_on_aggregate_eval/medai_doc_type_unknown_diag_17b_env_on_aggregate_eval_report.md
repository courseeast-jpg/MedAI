# MEDAI-DOC-TYPE-UNKNOWN-DIAG-17B Env-On Aggregate Evaluation

Reports-only / evaluation-only env-on aggregate evaluation of the DIAG-17 PDF text/layout quality metadata helper. No runtime behavior changes. No extraction behavior changes. PARK-20 / PARK-21 tags untouched.

## State

- Phase ID: `MEDAI-DOC-TYPE-UNKNOWN-DIAG-17B`
- Mode: `env_on_aggregate_evaluation`
- Evaluation only: **True**
- Reports only: **True**
- Env var evaluated: `MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_IMPL_ENABLED`
- Branch: `clinical-knowledge-architecture`
- HEAD commit (short): `6ed9962`
- DIAG-17 commit (short): `ad7b2d6`
- PARK-20 parking commit (short): `3e46461`
- PARK-21 parking commit (short): `9f9e22d`

## PARK status

PARK-20 tags remain on origin at 3e46461. PARK-21 tags remain on origin at 9f9e22d. DIAG-17B does not touch any tag.

## Source reports referenced

- `block DIAG-13A (directory: medai_doc_type_unknown_diag_13a_text_layer_diagnostic)`
- `block DIAG-14 (directory: medai_doc_type_unknown_diag_14_text_layer_extraction_spec)`
- `block DIAG-15 (directory: medai_doc_type_unknown_diag_15_pdf_text_extraction_quality_audit)`
- `block DIAG-15B (directory: medai_doc_type_unknown_diag_15b_layout_table_extraction_audit)`
- `block DIAG-16 (directory: medai_doc_type_unknown_diag_16_pdf_text_layout_quality_spec)`
- `block DIAG-17 (directory: medai_doc_type_unknown_diag_17_pdf_text_layout_quality_impl)`

## Env-on aggregate results

| Metric | Value |
| --- | ---: |
| `total_records_evaluated` | 21 |
| `emitted_metadata_count` | 21 |
| `suppressed_or_excluded_count` | 0 |
| `review_required_count` | 21 |
| `auto_accept_allowed_count` | 0 |
| `clinical_interpretation_performed_count` | 0 |
| `raw_text_emission_count` | 0 |
| `raw_filename_emission_count` | 0 |
| `private_path_emission_count` | 0 |
| `diagnosis_inference_count` | 0 |
| `medication_inference_count` | 0 |
| `ddi_inference_count` | 0 |
| `treatment_inference_count` | 0 |
| `abbreviation_expansion_count` | 0 |
| `external_api_used_count` | 0 |
| `park_20_tags_touched_count` | 0 |
| `park_21_tags_touched_count` | 0 |
| `excluded_pool_count` | 5 |
| `excluded_pool_emission_count` | 0 |

## Per-subtrack emit counts

| Sub-track | Emitted |
| --- | ---: |
| A | 11 |
| B | 10 |

## Family-label counts (multi-label, under env-on)

| Quality family label | Count |
| --- | ---: |
| `pdf_text_too_short` | 11 |
| `table_structure_visible_text_insufficient` | 10 |
| `layout_or_table_extraction_gap` | 10 |

## Default-off invariants AFTER evaluation

- Helper still default-disabled with env={}: **True**
- Helper still enabled only when env truthy: **True**

## Block invariants

- `behavior_changed`: False
- `runtime_behavior_changed`: False
- `extraction_behavior_changed`: False
- `pdf_text_extraction_behavior_changed`: False
- `layout_extraction_behavior_changed`: False
- `table_extraction_behavior_changed`: False
- `ocr_behavior_changed`: False
- `classifier_behavior_changed`: False
- `threshold_behavior_changed`: False
- `operator_ui_surface_added`: False
- `cue_expansion_recommended`: False
- `cue_expansion_performed`: False
- `external_api_used`: False
- `source_documents_opened`: False
- `raw_text_printed`: False
- `raw_filenames_printed`: False
- `private_paths_printed`: False
- `clinical_value_parsing_performed`: False
- `diagnosis_inference_performed`: False
- `medication_inference_performed`: False
- `ddi_inference_performed`: False
- `treatment_inference_performed`: False
- `abbreviation_expansion_performed`: False
- `park_20_tags_touched`: False
- `park_21_tags_touched`: False
- `all_records_review_bound`: True
- `accepted_count`: 0

## Safety / privacy

DIAG-17B is reports-only / evaluation-only. The DIAG-17 helper is exercised in explicit env-on mode via an in-process env mapping; os.environ is never written. No source documents, raw OCR text, raw document text, raw filenames, private paths, PHI, secrets, DBs, backups, or bundles are read or emitted. Output uses anonymized file_NNN IDs only. No runtime behavior changes. No extraction behavior changes. PARK-20 / PARK-21 tags are not touched.

## Progress estimate

| Track | Before | After |
| --- | --- | --- |
| Residual Unknown-reduction | ~99.95% done / ~0.05% remaining | ~99.97% done / ~0.03% remaining |
| Whole MedAI project | ~90.5% done / ~9.5% remaining | ~90.8% done / ~9.2% remaining |

## Recommended next block

- DIAG-18 — first env-gated, read-only operator surface for DIAG-17 metadata, behind a SEPARATE fourth env var. Default-off; no auto-accept; no clinical interpretation; no data-layer change. Mirrors the DIAG-08A / DIAG-10A / DIAG-12A pattern. Cue expansion remains explicitly NOT recommended.

