# MEDAI-PARK-22 — Short Summary

Reports-only and tag-only parking snapshot freezing the DIAG-17 + DIAG-17B
+ DIAG-18 default-off chain on origin.

## State

- Phase ID: `MEDAI-PARK-22`
- Mode: `parking_snapshot`
- Branch: `clinical-knowledge-architecture`
- HEAD before PARK-22: `9d7faec`
- PARK-20 parking commit: `3e46461` (tags untouched)
- PARK-21 parking commit: `9f9e22d` (tags untouched)
- Metadata env var: `MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_IMPL_ENABLED`
- UI env var: `MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_UI_ENABLED`
- Requires BOTH env vars truthy for the UI plan to render.

## Covered chain

- `MEDAI-DOC-TYPE-UNKNOWN-DIAG-17`
- `MEDAI-DOC-TYPE-UNKNOWN-DIAG-17B`
- `MEDAI-DOC-TYPE-UNKNOWN-DIAG-18`

## Tag plan

- `medai-pdf-text-layout-quality-default-off-ready-2026-05-19`
- `medai-final-parked-post-diag-18-2026-05-19`

Both annotated, both pointing at the PARK-22 commit, both pushed via the
github-direct route (the proxy still 403s on `refs/tags/*` writes).

## Top-level invariants

- `runtime_behavior_changed`: false
- `runtime_wiring_added`: false
- `streamlit_wiring_added`: false
- `extraction_behavior_changed`: false
- `ocr_behavior_changed`: false
- `classifier_behavior_changed`: false
- `threshold_behavior_changed`: false
- `cue_expansion_recommended`: false
- `external_api_used`: false
- `park_20_tags_touched`: false
- `park_21_tags_touched`: false
- `all_records_review_bound`: true
- `accepted_count`: 0
- `auto_accept_allowed_count`: 0
- `external_api_used_count`: 0

## Progress

- Residual Unknown-reduction track: ~99.98% done / ~0.02% remaining
- Whole MedAI project: ~91% done / ~9% remaining
- Release hygiene (post-PARK-22): 100% done / 0% remaining

## Recommended next block

DIAG-19 — env-gated Streamlit wiring of the DIAG-18 render plan into the
Run & Review tab's "Advanced technical details" expander, behind the
two-env-var gate. Default-off. No auto-accept. PARK-20 / PARK-21 / PARK-22
tags must remain untouched. Cue expansion remains explicitly NOT
recommended.
