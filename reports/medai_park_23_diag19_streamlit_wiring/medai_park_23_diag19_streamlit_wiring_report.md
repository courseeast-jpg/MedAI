# MEDAI-PARK-23 — Park DIAG-19 Env-Gated Streamlit Wiring

Reports-only and tag-only parking block. Freezes the
`clinical-knowledge-architecture` branch on origin after DIAG-19 wired the
DIAG-18 read-only render plan into `app/main.py::render_run_result_card`'s
Advanced technical details expander. No runtime behavior changes. No new
Streamlit wiring. No extraction / OCR / classifier / threshold / cue
changes. No external APIs. PARK-20, PARK-21, and PARK-22 tags must not be
touched.

## State

- Phase ID: `MEDAI-PARK-23`
- Mode: `parking_snapshot`
- Reports only: **true**
- Tags-only after commit: **true**
- Branch: `clinical-knowledge-architecture`
- Current HEAD before PARK-23: `d7c1db5`
- DIAG-19 implementation commit: `57c68b0`
- DIAG-19 receipt-refresh commit: `d7c1db5`
- PARK-20 parking commit: `3e46461`
- PARK-21 parking commit: `9f9e22d`
- PARK-22 parking commit: `f4d3cc6`
- Metadata env var: `MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_IMPL_ENABLED`
- UI env var: `MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_UI_ENABLED`
- Requires both env vars truthy for the wiring to render: **true**
- Default behavior changed: **false**

## PARK-20 / PARK-21 / PARK-22 tag status

- PARK-20 tags
  `medai-unknown-diag-language-metadata-ready-2026-05-19` and
  `medai-final-parked-post-unknown-diag-language-metadata-2026-05-19`
  remain on origin and resolve to `3e46461`.
- PARK-21 tags
  `medai-text-layer-eval-spec-ready-2026-05-19` and
  `medai-final-parked-post-diag-16-2026-05-19`
  remain on origin and resolve to `9f9e22d`.
- PARK-22 tags
  `medai-pdf-text-layout-quality-default-off-ready-2026-05-19` and
  `medai-final-parked-post-diag-18-2026-05-19`
  remain on origin and resolve to `f4d3cc6`.
- PARK-23 must not touch any of the six.

## Covered chain

| Block | Implementation commit | Receipt-refresh commit |
| --- | --- | --- |
| `MEDAI-DOC-TYPE-UNKNOWN-DIAG-19` | `57c68b0` | `d7c1db5` |

## 1. Why PARK-23 exists

DIAG-19 wired the DIAG-18 read-only render plan into
`app/main.py::render_run_result_card`'s Advanced technical details
expander, behind both the DIAG-17 metadata env var AND the DIAG-18 UI
env var. The wiring is default-off in every environment where either
flag is unset or falsy, and emits only safe controlled-vocabulary
markdown when both are truthy. PARK-23 freezes that wiring on origin
before any block that would extend the surface beyond read-only or flip
an env var on by default. It is the gating snapshot.

## 2. What DIAG-19 changed

- Added one optional wiring block in
  `app/main.py::render_run_result_card`, inside the existing Advanced
  technical details expander, immediately after the prior DIAG-08A /
  DIAG-10A / DIAG-12A optional blocks.
- The block uses a try/except guarded import for
  `clinical_knowledge.document_type.pdf_text_layout_quality_ui` and the
  `render_plan_for_pdf_text_layout_quality` helper.
- The block calls the helper with no kwargs, so the helper consults
  `os.environ` and returns `None` unless BOTH the DIAG-17 metadata env
  var AND the DIAG-18 UI env var are truthy.
- When the helper returns a plan, the block emits only `st.markdown`
  lines and one `st.caption` — no widgets, no buttons, no forms, no
  callbacks, no actions.
- No new operator UI route, no new tab, no new sidebar entry.

## 3. Why default behavior remains unchanged

- With no kwargs and the default empty mapping derived from
  `os.environ`, the DIAG-18 helper returns `None` whenever either env
  var is unset or falsy.
- The wiring block has an `if _pl_plan is not None:` guard, so when the
  helper returns `None` no Streamlit call executes.
- The try/except around the import keeps non-Streamlit environments
  importable without `ImportError`.
- No caller in any default-deployment configuration sets the DIAG-17
  metadata env var, the DIAG-18 UI env var, or both.
- DIAG-17 and DIAG-18 helpers continue to be default-off in the real
  `os.environ` (verified by DIAG-19 focused tests and the env-
  combination audit).

## 4. Two-env-var gate

| Env combination | Helper returns | Wiring renders |
| --- | :-: | :-: |
| Neither truthy | `None` | nothing |
| Metadata env truthy only | `None` | nothing |
| UI env truthy only | `None` | nothing |
| Both truthy | plan dict | controlled-vocabulary markdown + caption |

Gate enforced inside the DIAG-18 helper; the wiring does not add a
separate env check.

## 5. Read-only UI invariants

- `diag_19_block_appears_exactly_once_in_app_main`: true
- `inside_advanced_technical_details_expander`: true
- `after_prior_diag_08a_10a_12a_blocks`: true
- `uses_try_except_guarded_import`: true
- `uses_only_st_markdown_and_st_caption`: true
- `no_st_button_or_form_or_session_state_or_rerun`: true
- `no_on_click_on_change_on_submit_kwargs`: true
- `no_state_mutation_added`: true
- `no_data_layer_write_added`: true
- `no_document_type_mutation_added`: true
- `no_raw_text_or_filename_or_path_rendered`: true
- `no_park_tag_name_rendered`: true

## 6. Safety / privacy invariants

- `default_behavior_changed`: false
- `runtime_behavior_changed_by_default`: false
- `streamlit_wiring_added`: true
- `streamlit_wiring_enabled_by_default`: false
- `advanced_technical_details_only`: true
- `read_only`: true
- `buttons_added`: false
- `callbacks_added`: false
- `actions_added`: false
- `forms_added`: false
- `state_mutation_added`: false
- `data_layer_write_added`: false
- `document_type_mutation_added`: false
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
- `park_22_tags_touched`: false
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

## 7. Validation evidence (PARK-23 baseline)

| Validation | Result |
| --- | --- |
| DIAG-01..19 diagnostic suite | 1002 / 1002 passing |
| Document-type eval non-streamlit subset | 43 / 43 passing |
| Final CKA MVP validation | 693 tests / 26 preflight checks passing |
| B07 term01 opt-in integration | 6 / 6 passing |
| ROUTE-FIX 01 | passing |
| UI ops panel | passing |
| UI boot fix | passing |
| Public-report privacy checks | all DIAG-19 public reports pass `clinical_knowledge.privacy.check_public_report_payload` |
| Full repo-wide pytest | **Skipped.** Tests that import streamlit at module load fail collection in this sandbox; streamlit is not installed. The DIAG-19 wiring uses streamlit only via the existing module-level `import streamlit as st` in `app/main.py` and a try/except-guarded import of the DIAG-18 helper, so non-Streamlit test collection of the helper modules is unaffected. PARK-23 changes no runtime code, so a full re-run is unnecessary; per task instructions this skip is NOT counted as a regression failure. |

## 8. Tag plan

Two annotated tags are created at the PARK-23 commit:

- `medai-pdf-text-layout-quality-streamlit-wiring-ready-2026-05-19`
- `medai-final-parked-post-diag-19-2026-05-19`

Tag-push route: the local proxy historically returns HTTP 403 on
`refs/tags/*` writes, so tag pushes go through the github-direct route
(`https://github.com/courseeast-jpg/MedAI.git`) using the gh-installed
credentials configured earlier with explicit user approval.

Constraints:

- `--tags` flag must NOT be used.
- `--force` / `--force-with-lease` must NOT be used.
- No existing tag may be moved or deleted.
- PARK-20 tags must still resolve to `3e46461` after PARK-23 tags are
  pushed.
- PARK-21 tags must still resolve to `9f9e22d` after PARK-23 tags are
  pushed.
- PARK-22 tags must still resolve to `f4d3cc6` after PARK-23 tags are
  pushed.

## 9. Recommended next block after PARK-23

Either:

- (a) A corpus-side env-on operator UAT block (still default-off in
  production) that exercises the wired surface against the existing
  21-record text-layer scope; or
- (b) An evaluation-only block that audits the wiring under Streamlit
  fixture tests.

Both must remain default-off, review-bound, aggregate-only, and must not
touch PARK-20 / PARK-21 / PARK-22 / PARK-23 tags. Cue expansion remains
explicitly NOT recommended.

## Progress estimate

| Track | Before | After |
| --- | --- | --- |
| Residual Unknown-reduction | ~99.99% done / ~0.01% remaining | ~99.99% done / ~0.01% remaining |
| Whole MedAI project | ~91.5% done / ~8.5% remaining | ~91.5% done / ~8.5% remaining |
| Release hygiene (post-PARK-23) | snapshot pending | 100% done / 0% remaining |
