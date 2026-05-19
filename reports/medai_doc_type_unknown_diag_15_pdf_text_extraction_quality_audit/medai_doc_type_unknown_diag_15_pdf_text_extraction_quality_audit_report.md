# MEDAI-DOC-TYPE-UNKNOWN-DIAG-15 PDF Text-Extraction Quality Audit

Aggregate-only, evaluation-only audit of the 11 Sub-track A records from DIAG-14 (`text_layer_too_short`, no table structure, pdf_text_layer_detected=yes, image_like_pdf=no, alphabetic_content high). No runtime behavior changes. No extraction behavior changes. No source documents read.

## State

- Phase ID: `MEDAI-DOC-TYPE-UNKNOWN-DIAG-15`
- Mode: `evaluation_only`
- Aggregate only: **True**
- Sub-track: `A_pdf_text_extraction_quality_audit`
- Branch: `clinical-knowledge-architecture`
- HEAD commit (short): `b4db91d`
- DIAG-14 commit (short): `832a5fe`
- PARK-20 parking commit (short): `3e46461`

## PARK-20 tag status

PARK-20 branch parked at 3e46461. PARK-20 tags medai-unknown-diag-language-metadata-ready-2026-05-19 and medai-final-parked-post-unknown-diag-language-metadata-2026-05-19 exist on origin and resolve to 3e46461. Tags are not touched in DIAG-15.

## Source reports used

- `block DIAG-03 (directory: medai_doc_type_unknown_diag_03)`
- `block DIAG-13A (directory: medai_doc_type_unknown_diag_13a_text_layer_diagnostic)`
- `block DIAG-14 (directory: medai_doc_type_unknown_diag_14_text_layer_extraction_spec)`

## Total records analyzed: 11

Bucket semantics: multi-label flags per record-pool. A single record may satisfy multiple buckets at once.

## PDF text-extraction quality bucket counts

| Bucket | Count |
| --- | ---: |
| `text_layer_present_but_extracted_text_too_short` | 11 |
| `extraction_length_below_family_classifier_minimum` | 11 |
| `alphabetic_visibility_too_low` | 0 |
| `numeric_or_table_heavy_but_text_sparse` | 0 |
| `pdf_text_layer_detected_but_semantically_insufficient` | 11 |
| `possible_encoding_or_glyph_extraction_issue` | 0 |
| `possible_scanned_or_image_dominant_pdf_mislabeled_as_text_layer` | 0 |
| `metadata_sufficient_for_future_extraction_audit` | 11 |
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

DIAG-15 is a static, aggregate-only, evaluation-only audit of the 11 Sub-track A records first surfaced in DIAG-02, root-caused in DIAG-03, classified in DIAG-13A, and split out by DIAG-14. Bucket counts are multi-label flags derived from definitional properties of Sub-track A and from existing aggregate-only DIAG-03 signals. No source documents, raw OCR text, raw document text, raw filenames, private paths, PHI, secrets, DBs, backups, or bundles are read or emitted. Output uses anonymized file_NNN IDs only.

## Progress estimate

| Track | Before | After |
| --- | --- | --- |
| Residual Unknown-reduction | ~99% done / ~1% remaining | ~99.5% done / ~0.5% remaining |
| Whole MedAI project | ~88–89% done / ~11–12% remaining | ~89% done / ~11% remaining |

## Recommended next block

- DIAG-16 — PDF text-extraction quality spec (still evaluation-only, aggregate-only) OR DIAG-15B layout/table extraction audit for the 10 Sub-track B records. Cue expansion is explicitly NOT recommended as the primary next step.
- Must remain evaluation-only: **True**
- Must remain aggregate-only: **True**
- Must NOT propose cue expansion as primary step: **True**
- Must NOT change extraction behavior in first pass: **True**

