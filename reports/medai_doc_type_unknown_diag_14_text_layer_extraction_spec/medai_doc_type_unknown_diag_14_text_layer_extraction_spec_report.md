# MEDAI-DOC-TYPE-UNKNOWN-DIAG-14 Text-Layer Extraction Spec

Evaluation-only, aggregate-only specification for the 21 residual text-layer records. Splits them into two future audit sub-tracks (PDF text-extraction quality audit and layout/table extraction audit) and defines positive signatures, exclusion rules, future diagnostic behavior, future implementation acceptance criteria, future validation requirements, and rollback/safety boundaries.

## State

- Branch: `clinical-knowledge-architecture`
- HEAD commit (short): `13ae9ed`
- DIAG-13A commit (short): `7866ba4`
- PARK-20 parking commit (short): `3e46461`

## Remote tag caveat

Branch parked at PARK-20 (3e46461). PARK-20 tags exist locally and target 3e46461 but remote tag push is currently blocked by an HTTP 403 from origin's receive-pack endpoint. Tags are not touched in DIAG-14. The out-of-band GitHub/proxy permission fix remains pending.

## Source reports used

- `block DIAG-02 (directory: medai_doc_type_unknown_diag_02)`
- `block DIAG-03 (directory: medai_doc_type_unknown_diag_03)`
- `block DIAG-13-PREFLIGHT (directory: medai_doc_type_unknown_diag_13_preflight)`
- `block DIAG-13A (directory: medai_doc_type_unknown_diag_13a_text_layer_diagnostic)`

## Total text-layer records analyzed: 21

## Sub-track split

| Sub-track | Pool count |
| --- | ---: |
| A — PDF text-extraction quality audit | 11 |
| B — layout / table extraction audit | 10 |
| **Total** | **21** |

Split sum equals total: **True**

## Positive signature — sub-track A (PDF text-extraction quality audit)

- pdf_text_layer_detected = yes
- image_like_pdf = no
- text_layer_too_short = yes
- no extraction error reported in safe metadata
- no OCR routing required at the diagnostic stage
- no private or raw text required to perform the diagnosis
- record remains review-bound

## Positive signature — sub-track B (layout / table extraction audit)

- pdf_text_layer_detected = yes
- image_like_pdf = no
- table_structure_visible = yes
- table_structure_visible_but_text_insufficient = yes
- table-like shape present in safe metadata
- raw table contents not required in public outputs
- record remains review-bound

## Exclusion rules

- exclude image-like PDFs that would require OCR routing work
- exclude no-text-layer records (DIAG-02 no_text_layer subset)
- exclude fallback_ran_but_no_family_match records
- exclude ambiguous_below_threshold records
- exclude numeric-table safe-default records already handled by DIAG-06A/07A/08A
- exclude language-propagation records already handled by DIAG-09A/10A
- exclude latin-abbreviation records already handled by DIAG-11A/12A
- exclude the table-header special case (single deferred record)
- exclude records that would require lab-value parsing
- exclude records that would require medication / dose / frequency / duration / DDI parsing
- exclude records that would require abbreviation parsing or expansion
- exclude records with insufficient safe metadata for the chosen sub-track signature

## Proposed future diagnostic behavior

- the future block may audit extraction quality metadata only
- it may compare extraction length buckets, table-visibility buckets, and layout-signal buckets
- it must not output raw extracted text
- it must not parse clinical values
- it must not change OCR routing in the first diagnostic pass
- it must not change PDF text-extraction behavior in the first diagnostic pass
- it must not change layout/table extraction behavior in the first diagnostic pass
- it must preserve review-bound status for every affected record

## Future implementation acceptance criteria

- only exact-signature records are affected
- no auto-accept
- accepted_count remains 0
- auto_accept_allowed_count remains 0
- external_api_used_count remains 0
- all affected records remain review-bound
- data-layer Unknown behavior explicitly reported (changed vs unchanged)
- no false-positive expansion into treatment / imaging / administrative document types
- no raw text in public reports
- no raw filenames in public reports
- rollback / disable path exists if runtime changes are ever introduced

## Future validation requirements

- focused synthetic tests for both sub-tracks
- replay of the 21-record text-layer pool
- 507-file aggregate validation
- document-type eval regression tests
- public-report privacy checks on every new report
- final CKA MVP validation
- B07 validation
- ROUTE-FIX validation
- UI ops validation
- UI boot validation
- staged safety check

## Rollback / safety boundaries

- any runtime behavior introduced by a later block must be default-off
- any runtime behavior introduced by a later block must be env-gated by a SEPARATE env var
- any runtime behavior introduced by a later block must be removable by toggling its env var to a falsy value
- if any later block alters classifier, OCR routing, or extraction behavior outside its env-gated scope, the block is invalid and must be reverted
- no later block may emit raw text, raw filenames, or private paths to public reports

## Block invariants

- `behavior_changed`: False
- `external_api_used`: False
- `cue_expansion_recommended`: False
- `extraction_behavior_changed`: False
- `implementation_started`: False
- `runtime_helper_added`: False
- `operator_ui_surface_added`: False
- `ocr_routing_changed`: False
- `ocr_engine_behavior_changed`: False
- `pdf_text_extraction_behavior_changed`: False
- `layout_table_extraction_behavior_changed`: False
- `raw_language_detector_changed`: False
- `classifier_behavior_changed`: False
- `thresholds_or_scoring_changed`: False
- `cue_packs_added`: False
- `park_20_tags_touched`: False
- `no_extraction_behavior_implemented_in_this_block`: True
- `no_runtime_behavior_changed_in_this_block`: True

## Safety / privacy

DIAG-14 is a static, aggregate-only, evaluation-only specification block. It splits the 21 text-layer records into two future audit sub-tracks (11 + 10) and defines positive signatures, exclusion rules, future diagnostic behavior, future implementation acceptance criteria, future validation requirements, and rollback/safety boundaries. No source documents, raw OCR text, raw document text, raw filenames, private paths, PHI, secrets, DBs, backups, or bundles are read or emitted. Output uses anonymized file_NNN IDs only.

## Progress estimate

| Track | Before | After |
| --- | --- | --- |
| Residual Unknown-reduction | ~98% done / ~2% remaining | ~99% done / ~1% remaining |
| Whole MedAI project | ~87% done / ~13% remaining | ~88% done / ~12% remaining |

## Recommended next block

- DIAG-15 — first implementation pass for one of the two sub-tracks (PDF text-extraction quality audit OR layout/table extraction audit), strictly evaluation-only and aggregate-only at the diagnostic stage, mirroring the DIAG-02..05 pattern.
- Must remain evaluation-only: **True**
- Must remain aggregate-only: **True**
- Must NOT propose cue expansion as primary step: **True**
- Must NOT change extraction behavior in first pass: **True**

