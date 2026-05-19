# MEDAI-DOC-TYPE-UNKNOWN-DIAG-13A Text-Layer Diagnostic

Aggregate-only, evaluation-only diagnostic over the 21 residual `likely_text_layer_issue` records previously characterized by DIAG-02 and DIAG-03. No runtime behavior changes. No source documents read.

## State

- Branch: `clinical-knowledge-architecture`
- HEAD commit (short): `657e8c6`
- PARK-20 parking commit (short): `3e46461`
- DIAG-13 preflight commit (short): `01c2b69`

## Remote tag caveat

Branch parked at PARK-20 (3e46461). PARK-20 tags exist locally and target 3e46461 but remote tag push is currently blocked by an HTTP 403 from origin's receive-pack endpoint. Tags are not touched in DIAG-13A. The out-of-band GitHub/proxy permission fix remains pending.

## Source reports used

- `block DIAG-02 (directory: medai_doc_type_unknown_diag_02)`
- `block DIAG-03 (directory: medai_doc_type_unknown_diag_03)`
- `block DIAG-13-PREFLIGHT (directory: medai_doc_type_unknown_diag_13_preflight)`

## Total text-layer records analyzed: 21

## Root-cause candidate counts

| Bucket | Count |
| --- | ---: |
| `text_layer_too_short` | 11 |
| `table_structure_visible_but_text_insufficient` | 10 |
| `text_layer_present_but_low_signal` | 0 |
| `image_like_with_partial_text` | 0 |
| `no_safe_text_visibility_metadata` | 0 |
| `leave_manual_review` | 0 |

## Structural-shape counts

| Bucket | Count |
| --- | ---: |
| `table_like_structure_visible` | 10 |
| `row_or_column_pattern_visible` | 0 |
| `section_heading_shape_visible` | 0 |
| `lab_or_result_shape_possible` | 0 |
| `treatment_or_schedule_shape_possible` | 0 |
| `administrative_or_form_shape_possible` | 0 |
| `no_known_shape_visible` | 11 |

## Future-lever counts

| Bucket | Count |
| --- | ---: |
| `candidate_text_layer_extraction_diagnostic` | 21 |
| `candidate_pdf_text_extraction_quality_audit` | 11 |
| `candidate_layout_table_extraction_audit` | 10 |
| `candidate_manual_review_only` | 0 |
| `insufficient_metadata_for_next_action` | 0 |

## Raw signal counts (from DIAG-03)

- `alphabetic_content_bucket_counts`: high=21
- `image_like_pdf_counts`: no=21
- `native_text_length_bucket_counts`: none=10, short=8, tiny=3
- `pdf_text_layer_detected_counts`: yes=21
- `table_like_structure_detected_counts`: no=11, yes=10

## Future block decision

- Future implementation / spec block justified: **True**
- Recommended code: **A** — text_layer_extraction_diagnostic_spec
- Recommendation letter options:
  - **A**: text-layer extraction diagnostic/spec
  - **B**: PDF text-quality audit
  - **C**: layout/table extraction audit
  - **D**: leave manual review
  - **E**: insufficient metadata
- Sub-track split for recommendation A:
  - PDF text-extraction quality audit subset: 11
  - layout / table extraction audit subset: 10
- Cue expansion recommended as primary next step: **False**

## Block invariants

- `behavior_changed`: False
- `external_api_used`: False
- `cue_expansion_recommended`: False
- `implementation_started`: False
- `ocr_routing_changed`: False
- `ocr_engine_behavior_changed`: False
- `raw_language_detector_changed`: False
- `classifier_behavior_changed`: False
- `thresholds_or_scoring_changed`: False
- `cue_packs_added`: False
- `park_20_tags_touched`: False

## Safety / privacy

DIAG-13A is a static, aggregate-only, evaluation-only diagnostic. It re-classifies the 21 text-layer records previously characterized by DIAG-02 and DIAG-03 into three controlled-vocabulary axes. No source documents, raw OCR text, raw document text, raw filenames, private paths, PHI, secrets, DBs, backups, or bundles are read or emitted. Output uses anonymized file_NNN IDs only.

## Progress estimate

| Track | Before | After |
| --- | --- | --- |
| Residual Unknown-reduction | ~97% done / ~3% remaining | ~98% done / ~2% remaining |
| Whole MedAI project | ~86% done / ~14% remaining | ~87% done / ~13% remaining |

