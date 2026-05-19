# MEDAI-PARK-21 — Short Summary

Reports-only and tag-only parking snapshot freezing the text-layer
evaluation-only chain on origin.

## State

- Phase ID: `MEDAI-PARK-21`
- Mode: `parking_snapshot`
- Branch: `clinical-knowledge-architecture`
- HEAD before PARK-21: `b2ecde6`
- PARK-20 parking commit: `3e46461` (tags untouched)

## Covered chain

- `MEDAI-DOC-TYPE-UNKNOWN-DIAG-13A`
- `MEDAI-DOC-TYPE-UNKNOWN-DIAG-14`
- `MEDAI-DOC-TYPE-UNKNOWN-DIAG-15`
- `MEDAI-DOC-TYPE-UNKNOWN-DIAG-15B`
- `MEDAI-DOC-TYPE-UNKNOWN-DIAG-16`

## Tag plan

- `medai-text-layer-eval-spec-ready-2026-05-19`
- `medai-final-parked-post-diag-16-2026-05-19`

Both annotated, both pointing at the PARK-21 commit, both pushed via the
github-direct route (the proxy still 403s on `refs/tags/*` writes).

## Top-level invariants

- `runtime_behavior_changed`: false
- `extraction_behavior_changed`: false
- `ocr_behavior_changed`: false
- `classifier_behavior_changed`: false
- `threshold_behavior_changed`: false
- `cue_expansion_recommended`: false
- `external_api_used`: false
- `park_20_tags_touched`: false
- `all_records_review_bound`: true
- `accepted_count`: 0
- `auto_accept_allowed_count`: 0
- `external_api_used_count`: 0

## Progress

- Residual Unknown-reduction track: ~99.9% done / ~0.1% remaining
- Whole MedAI project: ~90% done / ~10% remaining
- Release hygiene (post-PARK-21): 100% done / 0% remaining

## Recommended next block

DIAG-17 — first env-gated implementation pass under DIAG-16 acceptance
criteria (default-off, separate env var, no auto-accept, aggregate-only,
zero regression).
