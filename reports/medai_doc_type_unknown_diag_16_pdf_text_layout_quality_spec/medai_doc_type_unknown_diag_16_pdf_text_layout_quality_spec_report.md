# MEDAI-DOC-TYPE-UNKNOWN-DIAG-16 Unified PDF Text + Layout/Table Extraction Quality Spec

Specification-only, evaluation-only, aggregate-only block. Consolidates DIAG-15 (Sub-track A) and DIAG-15B (Sub-track B) into a unified forward-looking spec. No runtime behavior changes. No extraction behavior changes.

## State

- Phase ID: `MEDAI-DOC-TYPE-UNKNOWN-DIAG-16`
- Mode: `specification_only`
- Evaluation only: **True**
- Aggregate only: **True**
- Branch: `clinical-knowledge-architecture`
- HEAD commit (short): `3e51e77`
- DIAG-14 commit (short): `832a5fe`
- DIAG-15 commit (short): `3e57ba7`
- DIAG-15B commit (short): `2e9b53b`
- PARK-20 parking commit (short): `3e46461`

## PARK-20 tag status

PARK-20 branch parked at 3e46461. PARK-20 tags medai-unknown-diag-language-metadata-ready-2026-05-19 and medai-final-parked-post-unknown-diag-language-metadata-2026-05-19 exist on origin and resolve to 3e46461. Tags are not touched in DIAG-16.

## Inputs

- `MEDAI-DOC-TYPE-UNKNOWN-DIAG-15`
- `MEDAI-DOC-TYPE-UNKNOWN-DIAG-15B`

## Source reports used

- `block DIAG-14 (directory: medai_doc_type_unknown_diag_14_text_layer_extraction_spec)`
- `block DIAG-15 (directory: medai_doc_type_unknown_diag_15_pdf_text_extraction_quality_audit)`
- `block DIAG-15B (directory: medai_doc_type_unknown_diag_15b_layout_table_extraction_audit)`

## Total records covered: **21** (11 Sub-track A + 10 Sub-track B)

## 1. Evidence summary from DIAG-15 and DIAG-15B

### Sub-track A — PDF text-extraction quality audit (`MEDAI-DOC-TYPE-UNKNOWN-DIAG-15`, 11 records)

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

### Sub-track B — layout/table extraction audit (`MEDAI-DOC-TYPE-UNKNOWN-DIAG-15B`, 10 records)

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

## 2. Unified problem statement

21 residual review-bound Unknown records have a present-but-too-short PDF text layer (image_like_pdf=no, pdf_text_layer_detected=yes, alphabetic_content=high). 11 records (DIAG-15 sub-track A) lack any table-like structure; the text layer itself is insufficient for the family classifier. 10 records (DIAG-15B sub-track B) carry a visible table-like structure whose extractable text is still insufficient for the family classifier. A unified extraction-quality strategy is needed that improves recoverable evidence for these 21 records without perturbing OCR routing, OCR engine behavior, PDF text-extraction behavior, layout/table extraction behavior, classifier behavior, thresholds, scoring, cue packs, or auto-accept logic.

## 3. Non-goals

- must not modify OCR routing
- must not modify OCR engine behavior
- must not modify PDF text-extraction behavior at this stage
- must not modify layout/table extraction behavior at this stage
- must not modify the language detector
- must not modify the classifier behavior for non-signature records
- must not modify thresholds or scoring
- must not add cue packs
- must not parse lab values
- must not parse medications, dose, frequency, duration, or DDI
- must not parse or expand abbreviations
- must not add clinical interpretation
- must not auto-accept any record
- must not promote any record's document type at the data layer
- must not emit raw text, raw filenames, or private paths in public reports
- must not enable any external API
- must not touch PARK-20 tags
- must not change B07, ROUTE-FIX, DB schema, or command allowlist

## 4. Future implementation acceptance criteria

- any future extraction-improvement block MUST be opt-in or env-gated
- any future block MUST preserve review-bound status for previously Unknown records unless a later explicit acceptance block authorizes otherwise
- any future block MUST NOT auto-accept
- any future block MUST NOT parse clinical values
- any future block MUST NOT infer diagnosis, medication, DDI, or treatment meaning
- any future block MUST produce privacy-safe aggregate reports only
- any future block MUST keep accepted_count = 0 for the 21-record scope
- any future block MUST keep auto_accept_allowed_count = 0 for the 21-record scope
- any future block MUST keep external_api_used_count = 0
- any future block MUST NOT change OCR routing in its first pass
- any future block MUST NOT change OCR engine behavior in its first pass
- any future block MUST NOT change classifier behavior in its first pass
- any future block MUST NOT change thresholds or scoring in its first pass
- any future block MUST prove zero regression in the DIAG-01..16 diagnostic suite
- any future block MUST prove zero regression in the document-type eval non-streamlit subset
- any future block MUST prove zero regression in the final CKA MVP validation
- any future block MUST prove zero regression in B07 term01
- any future block MUST prove zero regression in ROUTE-FIX 01
- any future block MUST prove zero regression in UI ops
- any future block MUST prove zero regression in UI boot
- any future block MUST pass public report privacy checks on every new report
- any future block MUST stage only its own scoped files; no source documents, PDFs, images, DOCX, runtime DBs, private corpus files, backups, bundles, keys, private files, or terminology data may be staged
- any future block MUST keep PARK-20 tags untouched
- any future block MUST use anonymized file_NNN IDs only in public reports

## 5. Rollback criteria

- if any future block's env-gated path is enabled, a single env-var flip MUST return the system to the default-off, pre-block runtime behavior
- if any DIAG-01..16 diagnostic regresses, the future block MUST be reverted before any further work
- if any operational validation (CKA MVP, B07, ROUTE-FIX, UI ops, UI boot) regresses, the future block MUST be reverted
- if any public report leaks raw text, raw filenames, private paths, PHI, or secrets, the future block MUST be reverted and the leaking artifact removed from history per repository policy
- if accepted_count, auto_accept_allowed_count, or external_api_used_count rise above zero for the 21-record scope without an explicit acceptance block, the future block MUST be reverted
- if review-bound status is lost for any of the 21 records without an explicit acceptance block, the future block MUST be reverted
- if any source document, PDF, image, DOCX, runtime DB, private corpus file, backup, bundle, key, private file, or terminology dataset is staged, the future block MUST be reverted and the staged artifact removed before any commit
- if any PARK-20 tag is moved, deleted, or repointed, the future block MUST be reverted and the tag restored to PARK-20 commit 3e46461

## 6. Privacy and safety gates

- every new public report MUST pass clinical_knowledge.privacy.check_public_report_payload
- every new public report MUST NOT contain raw OCR text, raw document text, raw filenames, or private paths
- every new public report MUST NOT contain PHI, secrets, API keys, tokens, or other credentials
- every new public report MUST emit anonymized file_NNN IDs only
- every new test MUST use synthetic safe inputs, never read real corpus files
- every new script MUST avoid reading raw source documents, raw OCR text, raw document text, raw filenames, terminology files, runtime DBs, backups, or bundles
- every new commit MUST stage only the block's own scoped files; receipt-refresh churn must be a separate housekeeping commit
- every push MUST go to the branch only; no tag is created, moved, or pushed by the future block

## 7. Required regression tests for future implementation

- DIAG-15 focused tests (Sub-track A audit) — must remain passing
- DIAG-15B focused tests (Sub-track B audit) — must remain passing
- DIAG-01 through DIAG-16 diagnostic suite — must remain passing
- document-type eval non-streamlit subset — must remain passing
- final CKA MVP validation — must remain passing
- B07 term01 opt-in integration — must remain passing
- ROUTE-FIX 01 — must remain passing
- UI ops panel validation — must remain passing
- UI boot fix validation — must remain passing
- public-report privacy checks on every new public report — must pass
- staged safety check — must show no source documents / no private corpus / no DBs / no backups / no bundles / no keys / no terminology data staged

## 8. Review-bound invariants

- all 21 affected records remain Needs review
- unknown_count for the 21-record scope is unchanged
- accepted_count for the 21-record scope is zero
- auto_accept_allowed_count for the 21-record scope is zero
- external_api_used_count is zero
- no record's data-layer document type is mutated by the future block
- no record's raw language-detector output is mutated by the future block
- no record's classifier output is mutated by the future block unless explicitly env-gated and within scope

## 9. Cue-expansion decision

Cue expansion is explicitly NOT recommended as the primary next step. The 21 records' shared failure mode is extracted-text insufficiency at the family-classifier input, not a missing cue. Adding cues without improving extracted-text quality would only widen false-positive risk across the rest of the corpus. The cue catalog therefore remains frozen for this scope until a future block first demonstrates evaluation-only that improved extracted-text quality has changed the shape of the residual gap.

## 10. Recommended next block

PARK-21 — parking snapshot capturing the full text-layer evaluation-only chain (DIAG-13A diagnostic, DIAG-14 sub-track split, DIAG-15 PDF text quality audit, DIAG-15B layout/table audit, DIAG-16 unified spec). Aggregate-only; reports-only; no implementation block begins before the snapshot is in place. Cue expansion still not recommended.

## Block invariants

- `behavior_changed`: False
- `external_api_used`: False
- `source_documents_opened`: False
- `raw_text_printed`: False
- `raw_filenames_printed`: False
- `private_paths_printed`: False
- `runtime_behavior_changed`: False
- `extraction_behavior_changed`: False
- `pdf_text_extraction_behavior_changed`: False
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

DIAG-16 is a specification-only, evaluation-only, aggregate-only block. It consolidates DIAG-15 (Sub-track A, 11 records) and DIAG-15B (Sub-track B, 10 records) into a unified forward specification covering future implementation acceptance criteria, rollback boundaries, privacy/safety gates, and required regression tests. No source documents, raw OCR text, raw document text, raw filenames, private paths, PHI, secrets, DBs, backups, or bundles are read or emitted. No runtime behavior changes. PARK-20 tags are not touched.

## Progress estimate

| Track | Before | After |
| --- | --- | --- |
| Residual Unknown-reduction | ~99.8% done / ~0.2% remaining | ~99.9% done / ~0.1% remaining |
| Whole MedAI project | ~89.5% done / ~10.5% remaining | ~90% done / ~10% remaining |

