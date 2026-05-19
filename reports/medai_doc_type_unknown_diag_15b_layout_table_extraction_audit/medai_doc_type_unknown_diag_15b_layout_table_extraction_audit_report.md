# MEDAI-DOC-TYPE-UNKNOWN-DIAG-15B Layout/Table Extraction Audit

Aggregate-only, evaluation-only audit of the 10 Sub-track B records from DIAG-14 (`table_structure_visible_but_text_insufficient`, pdf_text_layer_detected=yes, image_like_pdf=no, alphabetic_content high). No runtime behavior changes. No extraction behavior changes. No source documents read.

## State

- Phase ID: `MEDAI-DOC-TYPE-UNKNOWN-DIAG-15B`
- Mode: `evaluation_only`
- Aggregate only: **True**
- Sub-track: `B_layout_table_extraction_audit`
- Branch: `clinical-knowledge-architecture`
- HEAD commit (short): `1e7e2ad`
- DIAG-14 commit (short): `832a5fe`
- DIAG-15 commit (short): `3e57ba7`
- PARK-20 parking commit (short): `3e46461`

## PARK-20 tag status

PARK-20 branch parked at 3e46461. PARK-20 tags medai-unknown-diag-language-metadata-ready-2026-05-19 and medai-final-parked-post-unknown-diag-language-metadata-2026-05-19 exist on origin and resolve to 3e46461. Tags are not touched in DIAG-15B.

## Source reports used

- `block DIAG-03 (directory: medai_doc_type_unknown_diag_03)`
- `block DIAG-13A (directory: medai_doc_type_unknown_diag_13a_text_layer_diagnostic)`
- `block DIAG-14 (directory: medai_doc_type_unknown_diag_14_text_layer_extraction_spec)`
- `block DIAG-15 (directory: medai_doc_type_unknown_diag_15_pdf_text_extraction_quality_audit)`

## Total records analyzed: 10

Bucket semantics: multi-label flags per record-pool. A single record may satisfy multiple buckets at once.

## Layout/table extraction bucket counts

| Bucket | Count |
| --- | ---: |
| `table_structure_visible_but_text_insufficient` | 10 |
| `layout_structure_visible_but_family_classifier_input_sparse` | 10 |
| `row_or_column_structure_likely_lost` | 0 |
| `table_header_or_label_context_insufficient` | 10 |
| `numeric_or_grid_like_content_without_family_cues` | 10 |
| `possible_multi_column_or_fragmented_text_order_issue` | 0 |
| `possible_pdf_table_extraction_gap` | 10 |
| `metadata_sufficient_for_future_layout_audit` | 10 |
| `insufficient_safe_metadata` | 0 |

## Invariants

- `behavior_changed`: False
- `external_api_used`: False
- `source_documents_opened`: False
- `raw_text_printed`: False
- `raw_filenames_printed`: False
- `private_paths_printed`: False
- `runtime_behavior_changed`: False
- `extraction_behavior_changed`: False
- `layout_extraction_behavior_changed`: False
- `table_extraction_behavior_changed`: False
- `ocr_behavior_changed`: False
- `classifier_behavior_changed`: False
- `threshold_behavior_changed`: False
- `cue_expansion_recommended`: False
- `implementation_started`: False
- `runtime_helper_added`: False
- `operator_ui_surface_added`: False
- `park_20_tags_touched`: False
- `unknown_count_changed`: False
- `all_records_review_bound`: True
- `accepted_count`: 0
- `auto_accept_allowed_count`: 0
- `external_api_used_count`: 0

## Safety / privacy

DIAG-15B is a static, aggregate-only, evaluation-only audit of the 10 Sub-track B records first surfaced in DIAG-02, root-caused in DIAG-03, classified in DIAG-13A, and split out by DIAG-14. Bucket counts are multi-label flags derived from definitional properties of Sub-track B and from existing aggregate-only DIAG-03 signals. No source documents, raw OCR text, raw document text, raw filenames, private paths, PHI, secrets, DBs, backups, or bundles are read or emitted. Output uses anonymized file_NNN IDs only.

## Progress estimate

| Track | Before | After |
| --- | --- | --- |
| Residual Unknown-reduction | ~99.5% done / ~0.5% remaining | ~99.8% done / ~0.2% remaining |
| Whole MedAI project | ~89% done / ~11% remaining | ~89.5% done / ~10.5% remaining |

## Recommended next block

- DIAG-16 — unified PDF text + layout/table extraction quality spec covering both Sub-track A (11 records) and Sub-track B (10 records). Still evaluation-only, still aggregate-only. Cue expansion is explicitly NOT recommended as the primary next step.
- Must remain evaluation-only: **True**
- Must remain aggregate-only: **True**
- Must NOT propose cue expansion as primary step: **True**
- Must NOT change extraction behavior in first pass: **True**

