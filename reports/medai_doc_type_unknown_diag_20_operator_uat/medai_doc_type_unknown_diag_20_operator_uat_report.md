# MEDAI-DOC-TYPE-UNKNOWN-DIAG-20 Corpus-Side Env-On Operator UAT

Reports-only / aggregate-only env-on operator UAT exercising the DIAG-17 metadata helper, the DIAG-18 render-plan helper, and the DIAG-19 wiring contract (static audit) together. No runtime behavior changes. No extraction behavior changes. PARK-20 / PARK-21 / PARK-22 / PARK-23 tags untouched.

## State

- Phase ID: `MEDAI-DOC-TYPE-UNKNOWN-DIAG-20`
- Mode: `env_on_operator_uat`
- Reports only: **True**
- Aggregate only: **True**
- Branch: `clinical-knowledge-architecture`
- HEAD commit (short): `6ed9962`
- Metadata env var: `MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_IMPL_ENABLED` (DIAG-17)
- UI env var: `MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_UI_ENABLED` (DIAG-18)
- Requires BOTH env vars truthy for the wiring to render: **True**
- Env mapping only: **True**
- `os.environ` written: **False**

## Covered chain

- `MEDAI-DOC-TYPE-UNKNOWN-DIAG-17`
- `MEDAI-DOC-TYPE-UNKNOWN-DIAG-18`
- `MEDAI-DOC-TYPE-UNKNOWN-DIAG-19`

## Env-on UAT aggregate results (21-record cohort)

| Metric | Value |
| --- | ---: |
| `total_records_evaluated` | 21 |
| `emitted_metadata_count` | 21 |
| `emitted_render_plan_count` | 21 |
| `review_required_count` | 21 |
| `auto_accept_allowed_count` | 0 |
| `clinical_interpretation_performed_count` | 0 |
| `raw_text_emission_count` | 0 |
| `raw_filename_emission_count` | 0 |
| `private_path_emission_count` | 0 |
| `excluded_pool_count` | 5 |
| `excluded_pool_metadata_emission_count` | 0 |
| `excluded_pool_render_plan_count` | 0 |
| `forbidden_render_field_count` | 0 |

## Per-subtrack emit counts

| Sub-track | Render plans emitted |
| --- | ---: |
| A | 11 |
| B | 10 |

## Family-label counts (multi-label, under env-on)

| Quality family label | Count |
| --- | ---: |
| `pdf_text_too_short` | 11 |
| `table_structure_visible_text_insufficient` | 10 |
| `layout_or_table_extraction_gap` | 10 |

## Safe render-field counts (summed across plans)

| Field | Count |
| --- | ---: |
| `expander_label` | 21 |
| `markdown_lines` | 21 |
| `disclaimer_line` | 21 |
| `quality_family` | 21 |
| `env_vars` | 21 |
| `badge_vocab_token` | 21 |
| `badge_text` | 21 |
| `badge_source_block` | 21 |
| `is_read_only` | 21 |
| `is_review_bound` | 21 |
| `no_action_attached` | 21 |
| `no_button_attached` | 21 |
| `no_callback_attached` | 21 |
| `no_form_attached` | 21 |
| `no_state_mutation` | 21 |
| `no_data_layer_write` | 21 |
| `no_document_type_mutation` | 21 |

## Two-env-var gate verification

| Env combination | Emits a render plan |
| --- | :-: |
| Neither truthy | **False** |
| Metadata env truthy only | **False** |
| UI env truthy only | **False** |
| Both truthy | **True** |

## Default-off invariants AFTER UAT

- DIAG-17 metadata helper still default-disabled with env={}: **True**
- DIAG-18 UI helper still default-disabled with env={}: **True**
- `os.environ` still does not contain `MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_IMPL_ENABLED`: **True**
- `os.environ` still does not contain `MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_UI_ENABLED`: **True**

## DIAG-19 wiring block static audit

- Block span: lines `1154` – `1180` (27 lines)
- Has try/except guard: **True**
- Has DIAG-18 import inside try: **True**
- Has render-plan call inside try: **True**
- Unsafe `st.*` calls in block: `[]`
- Any forbidden `st.*` symbols: **False**
- Any forbidden kwargs: **False**
- Any forbidden tokens in code (comments stripped): **False**
- Any forbidden `on_*` / `write_*` prefixes in code: **False**
- PARK tag names referenced anywhere in block: **False**

## Block invariants

- `default_behavior_changed`: False
- `runtime_behavior_changed`: False
- `streamlit_wiring_changed`: False
- `extraction_behavior_changed`: False
- `pdf_text_extraction_behavior_changed`: False
- `layout_extraction_behavior_changed`: False
- `table_extraction_behavior_changed`: False
- `ocr_behavior_changed`: False
- `classifier_behavior_changed`: False
- `threshold_behavior_changed`: False
- `cue_expansion_recommended`: False
- `cue_expansion_performed`: False
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
- `park_20_tags_touched`: False
- `park_21_tags_touched`: False
- `park_22_tags_touched`: False
- `park_23_tags_touched`: False
- `all_records_review_bound`: True
- `accepted_count`: 0
- `auto_accept_allowed_count`: 0
- `external_api_used_count`: 0

## Safety / privacy

DIAG-20 is a reports-only / aggregate-only operator UAT. The DIAG-17 metadata helper and the DIAG-18 render-plan helper are exercised in explicit env-on mode via an in-process env mapping; os.environ is never written. The DIAG-19 wiring block is audited statically by AST + token-boundary scan and is never executed. No source documents, raw OCR text, raw document text, raw filenames, private paths, PHI, secrets, DBs, backups, or bundles are read or emitted. Output uses anonymized file_NNN IDs only. No runtime behavior changes. No extraction behavior changes. PARK-20 / PARK-21 / PARK-22 / PARK-23 tags are not touched.

## Progress estimate

| Track | Before | After |
| --- | --- | --- |
| Residual Unknown-reduction | ~99.99% done / ~0.01% remaining | ~99.995% done / ~0.005% remaining |
| Whole MedAI project | ~91.5% done / ~8.5% remaining | ~91.7% done / ~8.3% remaining |

## Recommended next block

- Either (a) PARK-24 — a parking snapshot of the DIAG-20 UAT receipt, mirroring PARK-22 / PARK-23 (reports-only / tags-only; no runtime change), or (b) DIAG-21 — a Streamlit fixture-test audit of the DIAG-19 wiring block (reports-only; install streamlit in a controlled test env or use a mocking fixture pattern). Both must remain default-off, review-bound, aggregate-only, and must not touch PARK-20 / PARK-21 / PARK-22 / PARK-23 tags. Cue expansion remains explicitly NOT recommended.

