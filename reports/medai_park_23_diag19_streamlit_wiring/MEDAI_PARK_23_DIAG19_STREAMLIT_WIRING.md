# MEDAI-PARK-23 — Short Summary

Reports-only and tag-only parking snapshot freezing the DIAG-19
env-gated Streamlit wiring on origin.

## State

- Phase ID: `MEDAI-PARK-23`
- Mode: `parking_snapshot`
- Branch: `clinical-knowledge-architecture`
- HEAD before PARK-23: `d7c1db5`
- DIAG-19 commit: `57c68b0`
- DIAG-19 receipt-refresh commit: `d7c1db5`
- PARK-20 parking commit: `3e46461` (tags untouched)
- PARK-21 parking commit: `9f9e22d` (tags untouched)
- PARK-22 parking commit: `f4d3cc6` (tags untouched)
- Metadata env var: `MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_IMPL_ENABLED`
- UI env var: `MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_UI_ENABLED`
- Requires BOTH env vars truthy for the wiring to render.

## Covered chain

- `MEDAI-DOC-TYPE-UNKNOWN-DIAG-19`

## Tag plan

- `medai-pdf-text-layout-quality-streamlit-wiring-ready-2026-05-19`
- `medai-final-parked-post-diag-19-2026-05-19`

Both annotated, both pointing at the PARK-23 commit, both pushed via
the github-direct route (the proxy still 403s on `refs/tags/*` writes).

## Top-level invariants

- `default_behavior_changed`: false
- `runtime_behavior_changed_by_default`: false
- `streamlit_wiring_added`: true
- `streamlit_wiring_enabled_by_default`: false
- `advanced_technical_details_only`: true
- `read_only`: true
- `buttons_added` / `callbacks_added` / `actions_added` / `forms_added` /
  `state_mutation_added` / `data_layer_write_added` /
  `document_type_mutation_added`: false
- `extraction_behavior_changed` / `ocr_behavior_changed` /
  `classifier_behavior_changed` / `threshold_behavior_changed`: false
- `cue_expansion_recommended`: false
- `external_api_used`: false
- `park_20_tags_touched`: false
- `park_21_tags_touched`: false
- `park_22_tags_touched`: false
- `all_records_review_bound`: true
- `accepted_count`: 0
- `auto_accept_allowed_count`: 0
- `external_api_used_count`: 0

## Progress

- Residual Unknown-reduction track: ~99.99% done / ~0.01% remaining
- Whole MedAI project: ~91.5% done / ~8.5% remaining
- Release hygiene (post-PARK-23): 100% done / 0% remaining

## Recommended next block

Either (a) a corpus-side env-on operator UAT block (still default-off
in production) that exercises the wired surface against the 21-record
text-layer scope, or (b) an evaluation-only block that audits the
wiring under Streamlit fixture tests. Both must remain default-off,
review-bound, aggregate-only, and must not touch PARK-20 / PARK-21 /
PARK-22 / PARK-23 tags. Cue expansion remains explicitly NOT
recommended.
