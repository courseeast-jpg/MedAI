# MEDAI-AI-FIRST-CORPUS-17C-R2-FINAL-CLOSURE-SNAPSHOT-01

Read-only closure snapshot for Corpus 1 / 17C-R2 after R15. This document records the
final state and the closure decision. It performs no provider/live call, opens no MKB, and
commits no private artifact.

## Final state (R15)
- result: **LIVE_FAIL**
- HEAD after R15 public report commit: `b3ff56bb-76d2e4e6-b022ee42-2a3272e9-268f3b58`
  (hyphens are 8-char grouping separators; remove them for the canonical 40-char SHA)
- docs loaded / completed / failed-for-review / unattempted: **478 / 118 / 360 / 0**
- failure stage / category: **provider_live_fail / api_disabled_or_permission**
- actual total token count: **2,504,552**
- actual public cost: **$0.383599** (cap **$10.00**; per-chunk cap **$0.05**)
- 17D MKB import started: **false**
- privacy / safety: **passed**; public PHI / path / secret leaks: **0 / 0 / 0**

## Closure decision
**stop_live_retry_preserve_completed_and_failed_for_review.**

The 118 completed AI packages and the 360 failed-for-review evidence sets are preserved
privately. No retries are run; no MKB import is started. The blocker is provider/cloud side
(API disabled or insufficient permission), so it is cleared by a cloud/permission change
rather than by re-running the extraction code.

## Reusable assets
The full 17C-R2 architecture (strict-JSON contract, required-key skeleton, checkpoint /
resume, failed-evidence preservation, public-report redaction, raised output ceiling +
compact output, adaptive cost/chunk planning, sectioned extraction, autonomous recovery)
is reusable unchanged for Corpus 2. See
`reports/medai_ai_first_corpus_17c_r2_final_closure_snapshot_01/reusable_architecture_assets_public.md`.

## Pointers
- Reports: `reports/medai_ai_first_corpus_17c_r2_final_closure_snapshot_01/`
- Design index: `docs/pilot_design/MEDAI_AI_FIRST_CORPUS_17C_R2_FINAL_CLOSURE_SNAPSHOT_01/`
- Next-steps policy:
  `reports/medai_ai_first_corpus_17c_r2_final_closure_snapshot_01/next_steps_policy_public.md`

Corpus 1 is closed in this state. Corpus 2 preparation proceeds independently in its own
repository and branch and is not merged into `clinical-knowledge-architecture`.
