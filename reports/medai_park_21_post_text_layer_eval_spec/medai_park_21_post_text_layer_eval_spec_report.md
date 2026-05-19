# MEDAI-PARK-21 — Post Text-Layer Evaluation-Only Parking Snapshot

Reports-only and tag-only parking block. Freezes the
`clinical-knowledge-architecture` branch on origin after the completion of
the text-layer evaluation-only chain (DIAG-13A → DIAG-14 → DIAG-15 →
DIAG-15B → DIAG-16). No runtime behavior changes. No extraction behavior
changes. No OCR behavior changes. No classifier behavior changes. No
threshold/scoring changes. No cue expansion. No operator UI changes. No
external APIs. PARK-20 tags untouched.

## State

- Phase ID: `MEDAI-PARK-21`
- Mode: `parking_snapshot`
- Reports only: **true**
- Branch: `clinical-knowledge-architecture`
- Current HEAD before PARK-21: `b2ecde6`
- PARK-20 parking commit: `3e46461`

## PARK-20 tag status

PARK-20 tags `medai-unknown-diag-language-metadata-ready-2026-05-19` and
`medai-final-parked-post-unknown-diag-language-metadata-2026-05-19` remain
on origin and resolve to `3e46461`. PARK-21 must not touch them.

## Covered chain

| Block | Implementation commit | Receipt-refresh commit |
| --- | --- | --- |
| `MEDAI-DOC-TYPE-UNKNOWN-DIAG-13A` | `7866ba4` | `13ae9ed` |
| `MEDAI-DOC-TYPE-UNKNOWN-DIAG-14` | `832a5fe` | `b4db91d` |
| `MEDAI-DOC-TYPE-UNKNOWN-DIAG-15` | `3e57ba7` | `1e7e2ad` |
| `MEDAI-DOC-TYPE-UNKNOWN-DIAG-15B` | `2e9b53b` | `3e51e77` |
| `MEDAI-DOC-TYPE-UNKNOWN-DIAG-16` | `e144376` | `b2ecde6` |

## 1. Why PARK-21 exists

DIAG-13A through DIAG-16 produced a fully evaluation-only, aggregate-only
characterization of the 21 residual text-layer Unknown records:
classification (DIAG-13A), sub-track split (DIAG-14: 11 PDF-text +
10 layout/table), per-sub-track audits (DIAG-15 and DIAG-15B), and a
unified forward specification (DIAG-16). PARK-21 freezes that state on
origin before any block that introduces runtime extraction behavior begins.
It is the gating snapshot.

## 2. What the text-layer evaluation-only chain established

- 21 residual review-bound Unknown records share the failure mode "PDF
  text layer present but extracted text insufficient for family
  classification".
- Sub-track A (11 records): no table-like structure, text layer too short.
- Sub-track B (10 records): table-like structure visible but extractable
  text still insufficient.
- All 21 records have `pdf_text_layer_detected=yes`, `image_like_pdf=no`,
  `alphabetic_content=high`.
- Cue expansion is the wrong lever for this scope; the gap is extracted-
  text quality at the family-classifier input, not a missing cue.
- DIAG-16 fixed the future-implementation acceptance criteria, rollback
  boundaries, privacy / safety gates, and required regression tests.

## 3. What did not change

- OCR routing.
- OCR engine behavior.
- PDF text-extraction behavior.
- Layout / table extraction behavior.
- Raw language detector behavior.
- Classifier behavior.
- Thresholds or scoring.
- Cue packs.
- B07, ROUTE-FIX, DB schema, command allowlist.
- External API behavior.
- Operator UI surfaces.
- Runtime helpers.
- Lab value parsing.
- Medication / dose / frequency / duration / DDI parsing.
- Abbreviation parsing or expansion.
- Clinical interpretation.
- Data-layer document type for any of the 21 records.
- PARK-20 tags.

## 4. Safety / privacy invariants

- `runtime_behavior_changed`: false
- `extraction_behavior_changed`: false
- `pdf_text_extraction_behavior_changed`: false
- `layout_extraction_behavior_changed`: false
- `table_extraction_behavior_changed`: false
- `ocr_behavior_changed`: false
- `classifier_behavior_changed`: false
- `threshold_behavior_changed`: false
- `cue_expansion_recommended`: false
- `external_api_used`: false
- `external_api_enabled`: false
- `source_documents_opened`: false
- `source_documents_staged`: false
- `private_files_staged`: false
- `raw_text_printed`: false
- `raw_filenames_printed`: false
- `private_paths_printed`: false
- `raw_ocr_text_in_public_reports`: false
- `raw_document_text_in_public_reports`: false
- `raw_filenames_in_public_reports`: false
- `private_paths_in_public_reports`: false
- `secrets_in_public_reports`: false
- `park_20_tags_touched`: false
- `implementation_started`: false
- `runtime_helper_added`: false
- `operator_ui_surface_added`: false

Review-bound invariants:

- `unknown_count_changed`: false
- `accepted_count`: 0
- `auto_accept_allowed_count`: 0
- `external_api_used_count`: 0
- `all_records_review_bound`: true

## 5. Validation evidence (PARK-21 baseline)

| Validation | Result |
| --- | --- |
| DIAG-01..16 diagnostic suite | 835 / 835 passing |
| Document-type eval non-streamlit subset | 43 / 43 passing |
| Final CKA MVP validation | 693 tests / 26 preflight checks passing |
| B07 term01 opt-in integration | 6 / 6 passing |
| ROUTE-FIX 01 | passing |
| UI ops panel | passing |
| UI boot fix | passing |
| Public-report privacy checks | all DIAG-13A..16 public reports pass `clinical_knowledge.privacy.check_public_report_payload` |
| Full repo-wide pytest | **Skipped.** Tests that import streamlit at module load fail collection in this sandbox; streamlit is not installed. PARK-21 changes no runtime code, so a full re-run is unnecessary; per task instructions this skip is NOT counted as a regression failure. |

## 6. Tag plan

Two annotated tags are created at the PARK-21 commit:

- `medai-text-layer-eval-spec-ready-2026-05-19`
- `medai-final-parked-post-diag-16-2026-05-19`

Tag-push route: the local proxy historically returns HTTP 403 on
`refs/tags/*` writes, so tag pushes go through the github-direct route
(`https://github.com/courseeast-jpg/MedAI.git`) using the gh-installed
credentials configured earlier with explicit user approval.

Constraints:

- `--tags` flag must NOT be used.
- `--force` / `--force-with-lease` must NOT be used.
- No existing tag may be moved or deleted.
- PARK-20 tags must still resolve to `3e46461` after PARK-21 tags are
  pushed.

## 7. Recommended next block after PARK-21

**DIAG-17 — first env-gated implementation pass under DIAG-16 acceptance
criteria.** Default-off; behind a SEPARATE env var; no auto-accept; no
clinical parsing; aggregate-only reports; zero regression across
DIAG-01..16 plus all operational validations (CKA MVP, B07, ROUTE-FIX,
UI ops, UI boot). Cue expansion remains explicitly NOT recommended.

## Progress estimate

| Track | Before | After |
| --- | --- | --- |
| Residual Unknown-reduction | ~99.9% done / ~0.1% remaining | ~99.9% done / ~0.1% remaining |
| Whole MedAI project | ~90% done / ~10% remaining | ~90% done / ~10% remaining |
| Release hygiene (post-PARK-21) | snapshot pending | 100% done / 0% remaining |
