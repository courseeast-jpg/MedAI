# MEDAI-PARK-22 — Park PDF Text/Layout Quality Default-Off Chain

Reports-only and tag-only parking block. Freezes the
`clinical-knowledge-architecture` branch on origin after the completion
of the DIAG-17 + DIAG-17B + DIAG-18 default-off chain. No runtime
behavior changes. No Streamlit wiring. No extraction / OCR / classifier
/ threshold / cue changes. No external APIs. PARK-20 and PARK-21 tags
must not be touched.

## State

- Phase ID: `MEDAI-PARK-22`
- Mode: `parking_snapshot`
- Reports only: **true**
- Branch: `clinical-knowledge-architecture`
- Current HEAD before PARK-22: `9d7faec`
- PARK-20 parking commit: `3e46461`
- PARK-21 parking commit: `9f9e22d`
- Metadata env var: `MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_IMPL_ENABLED`
- UI env var: `MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_UI_ENABLED`
- Requires both env vars truthy for the UI plan to render: **true**
- Default behavior changed: **false**

## PARK-20 / PARK-21 tag status

- PARK-20 tags `medai-unknown-diag-language-metadata-ready-2026-05-19` and
  `medai-final-parked-post-unknown-diag-language-metadata-2026-05-19` remain
  on origin and resolve to `3e46461`.
- PARK-21 tags `medai-text-layer-eval-spec-ready-2026-05-19` and
  `medai-final-parked-post-diag-16-2026-05-19` remain on origin and resolve
  to `9f9e22d`.
- PARK-22 must not touch either pair.

## Covered chain

| Block | Implementation commit | Receipt-refresh commit |
| --- | --- | --- |
| `MEDAI-DOC-TYPE-UNKNOWN-DIAG-17` | `ad7b2d6` | `3adb32f` |
| `MEDAI-DOC-TYPE-UNKNOWN-DIAG-17B` | `3bc8a64` | `212f73c` |
| `MEDAI-DOC-TYPE-UNKNOWN-DIAG-18` | `f3c9760` | `9d7faec` |

## 1. Why PARK-22 exists

DIAG-17 added a strictly default-off PDF text/layout quality metadata
helper behind a SEPARATE new env var. DIAG-17B characterized its env-on
emission shape against the 21-record text-layer scope and confirmed zero
auto-accept / zero external API / zero clinical interpretation. DIAG-18
added a strictly default-off env-gated read-only operator render-plan
helper requiring BOTH the DIAG-17 metadata env var AND a NEW DIAG-18 UI
env var to be truthy. PARK-22 freezes that three-block default-off chain
on origin before any block that wires DIAG-18 into a Streamlit runtime
path or flips an env var on by default. It is the gating snapshot.

## 2. What DIAG-17 added

- New helper module `clinical_knowledge/document_type/pdf_text_layout_quality_impl.py`.
- Pure function `derive_pdf_text_layout_quality_context(record, *, enabled=None, env=None)`.
- New SEPARATE env var `MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_IMPL_ENABLED`.
- Default-off semantics: returns `None` when the env var is unset or falsy.
- Controlled-vocabulary `quality_family` values (`pdf_text_too_short`,
  `table_structure_visible_text_insufficient`,
  `layout_or_table_extraction_gap`).
- No runtime caller; not imported by `app/main.py`; not re-exported from
  the package `__init__.py`.

## 3. What DIAG-17B proved

- Env-on emission count equals **21** for the in-scope cohort.
- Excluded-pool emission count equals **0**.
- `auto_accept_allowed_count` = **0** under env-on.
- `external_api_used_count` = **0** under env-on.
- `clinical_interpretation_performed_count` = **0** under env-on.
- `diagnosis_inference_count` / `medication_inference_count` /
  `ddi_inference_count` / `treatment_inference_count` all = **0**.
- `abbreviation_expansion_count` = **0**.
- `raw_text_emission_count` / `raw_filename_emission_count` /
  `private_path_emission_count` all = **0**.
- `park_20_tags_touched_count` and `park_21_tags_touched_count` both = **0**.
- Helper remains default-disabled after evaluation (`os.environ` not
  written; in-process env mapping only).

## 4. What DIAG-18 added

- New helper module `clinical_knowledge/document_type/pdf_text_layout_quality_ui.py`.
- Pure function `render_plan_for_pdf_text_layout_quality(record, *, enabled=None, env=None)`.
- New SEPARATE UI env var `MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_UI_ENABLED`.
- Strict two-key env gate: BOTH the DIAG-17 metadata env var AND the
  DIAG-18 UI env var must be truthy for a plan to render.
- Pure data-only render plan; no Streamlit widgets, no buttons, no
  callbacks, no actions, no state mutation, no data-layer writes, no
  document_type mutation.
- Controlled-vocabulary expander label, badge text, badge vocab token,
  disclaimer, and per-record `quality_family` descriptions.
- No runtime caller; not imported by `app/main.py`; not re-exported from
  the package `__init__.py`. Zero Streamlit imports in the module.

## 5. What remains default-off

- DIAG-17 metadata helper (default-off; returns `None` unless the DIAG-17
  metadata env var is truthy).
- DIAG-18 operator render-plan helper (default-off; returns `None` unless
  BOTH env vars are truthy).
- No Streamlit wiring exists for either helper.
- No runtime call path invokes either helper by default.

## 6. What did not change

- OCR routing.
- OCR engine behavior.
- Default PDF text-extraction behavior.
- Default layout/table extraction behavior.
- Raw language detector behavior.
- Classifier behavior.
- Thresholds or scoring.
- Cue packs.
- B07, ROUTE-FIX, DB schema, command allowlist.
- External API behavior.
- Operator UI surfaces in `app/main.py`.
- Lab value parsing.
- Medication / dose / frequency / duration / DDI parsing.
- Abbreviation parsing or expansion.
- Clinical interpretation.
- Data-layer document type for any of the 21 in-scope records.
- PARK-20 tags.
- PARK-21 tags.

## 7. Safety / privacy invariants

- `runtime_behavior_changed`: false
- `runtime_wiring_added`: false
- `streamlit_wiring_added`: false
- `extraction_behavior_changed`: false
- `pdf_text_extraction_behavior_changed`: false
- `layout_extraction_behavior_changed`: false
- `table_extraction_behavior_changed`: false
- `ocr_behavior_changed`: false
- `classifier_behavior_changed`: false
- `threshold_behavior_changed`: false
- `cue_expansion_recommended`: false
- `cue_expansion_performed`: false
- `external_api_used`: false
- `external_api_enabled`: false
- `source_documents_opened`: false
- `source_documents_staged`: false
- `private_files_staged`: false
- `raw_text_printed`: false
- `raw_filenames_printed`: false
- `private_paths_printed`: false
- `raw_text_rendered`: false
- `raw_filenames_rendered`: false
- `private_paths_rendered`: false
- `raw_ocr_text_in_public_reports`: false
- `raw_document_text_in_public_reports`: false
- `raw_filenames_in_public_reports`: false
- `private_paths_in_public_reports`: false
- `secrets_in_public_reports`: false
- `park_20_tags_touched`: false
- `park_21_tags_touched`: false
- `clinical_value_parsing_performed`: false
- `diagnosis_inference_performed`: false
- `medication_inference_performed`: false
- `ddi_inference_performed`: false
- `treatment_inference_performed`: false
- `abbreviation_expansion_performed`: false

Review-bound invariants:

- `accepted_count`: 0
- `auto_accept_allowed_count`: 0
- `external_api_used_count`: 0
- `all_records_review_bound`: true

## 8. Validation evidence (PARK-22 baseline)

| Validation | Result |
| --- | --- |
| DIAG-01..18 diagnostic suite | 973 / 973 passing |
| Document-type eval non-streamlit subset | 43 / 43 passing |
| Final CKA MVP validation | 693 tests / 26 preflight checks passing |
| B07 term01 opt-in integration | 6 / 6 passing |
| ROUTE-FIX 01 | passing |
| UI ops panel | passing |
| UI boot fix | passing |
| Public-report privacy checks | all DIAG-17 / DIAG-17B / DIAG-18 public reports pass `clinical_knowledge.privacy.check_public_report_payload` |
| Full repo-wide pytest | **Skipped.** Tests that import streamlit at module load fail collection in this sandbox; streamlit is not installed. PARK-22 changes no runtime code, so a full re-run is unnecessary; per task instructions this skip is NOT counted as a regression failure. |

## 9. Tag plan

Two annotated tags are created at the PARK-22 commit:

- `medai-pdf-text-layout-quality-default-off-ready-2026-05-19`
- `medai-final-parked-post-diag-18-2026-05-19`

Tag-push route: the local proxy historically returns HTTP 403 on
`refs/tags/*` writes, so tag pushes go through the github-direct route
(`https://github.com/courseeast-jpg/MedAI.git`) using the gh-installed
credentials configured with explicit user approval.

Constraints:

- `--tags` flag must NOT be used.
- `--force` / `--force-with-lease` must NOT be used.
- No existing tag may be moved or deleted.
- PARK-20 tags must still resolve to `3e46461` after PARK-22 tags are
  pushed.
- PARK-21 tags must still resolve to `9f9e22d` after PARK-22 tags are
  pushed.

## 10. Recommended next block after PARK-22

**DIAG-19 — env-gated Streamlit wiring of the DIAG-18 render plan into
the Run & Review tab's "Advanced technical details" expander.** Mirrors
the DIAG-08A / DIAG-10A / DIAG-12A wiring shape but behind the two-env-var
gate. Default-off. No auto-accept. No clinical interpretation. PARK-20,
PARK-21, and PARK-22 tags must remain untouched. Cue expansion remains
explicitly NOT recommended.

## Progress estimate

| Track | Before | After |
| --- | --- | --- |
| Residual Unknown-reduction | ~99.98% done / ~0.02% remaining | ~99.98% done / ~0.02% remaining |
| Whole MedAI project | ~91% done / ~9% remaining | ~91% done / ~9% remaining |
| Release hygiene (post-PARK-22) | snapshot pending | 100% done / 0% remaining |
