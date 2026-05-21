# MEDAI-ROADMAP-06 — Post-PARK-02 Strategic Re-Evaluation

Reports-only planning audit. No runtime change. No `app/main.py` /
launcher / preflight / config / helper / Streamlit-wiring
modification. No tags created. PARK-20..23 tag pairs, PARK-24 / PARK-25
single tags, PARK-26 tag pair, the FREEZE tag pair, the term helper/
wiring PARK-01 tag pair, and the new license-gate PARK-02 tag pair all
remain intact. Cue expansion remains explicitly **NOT** recommended.

## 1. Executive summary

The honest default after PARK-02 is **FREEZE-MAINTENANCE-ONLY (A)**.
Every expansion track is now parked, frozen, or operator-side-blocked.
No fresh failure signal is driving forward motion. If forward motion
is desired, the safest exit ramp is **V2-ARCHITECTURE-SPEC-01** — a
reports-only architecture spec with no runtime change and no license
exposure.

| Field | Value |
| --- | :-: |
| `current_project_status` | **stable_frozen_local_operator_release_with_parked_expansion_tracks** |
| `top_recommended_next_phase` | **FREEZE-MAINTENANCE-ONLY** |
| Recommended next-3 sequence | A → V2-ARCHITECTURE-SPEC-01 → ROADMAP-07 |
| `license_gated_resources_verified_count` | 0 / 8 |
| `internal_boundaries_verified_count` | 2 / 2 |
| `manual_license_verification_complete` | false |
| `private_adapter_implementation_allowed` | false |
| `mesh_status` | `download_helper_created` |
| `cue_expansion_recommended` | **false** |

## 2. Why ROADMAP-06 exists

`MEDAI-CKA-TERM-INTEGRATION-PARK-02` (commit `b9b19ad`) parked the
license-gated private adapter track and recommended
`ROADMAP-06_or_freeze_maintenance`. With every committed expansion
track now parked or frozen, ROADMAP-06 records the durable inventory
and either selects the next safe forward-motion candidate or formally
chooses the freeze-maintenance default.

## 3. Current frozen / parked baseline

Pulled from the public-safe parking and freeze reports:

| Signal | Value |
| --- | :-: |
| Branch | `clinical-knowledge-architecture` |
| HEAD before this block | `358bcd4` (PARK-02 receipt refresh) |
| Freeze commit | `7ef8ffd` |
| `local_operator_release_frozen` | true (since `7ef8ffd`) |
| `cumulative_runtime_behavior_changed_across_recent_chain` | false |
| `cue_expansion_recommended` | false |
| `external_api_used` / `external_api_enabled` | false / false |
| Whole MedAI project | **~96.0%** done / ~4.0% remaining |

## 4. Parked track inventory

| Track | Status | Latest commit | Tags | `next_allowed_action` |
| --- | --- | --- | --- | --- |
| Local operator release freeze | **frozen** | `7ef8ffd` | FREEZE pair | maintenance-only re-runs |
| Residual Unknown diagnostics | parked | `3e46461` | PARK-20 pair | deferred (fresh-UAT-only) |
| PDF text/layout quality track | parked | `748c32a` | PARK-23 pair | deferred (fresh-UAT-only) |
| DIAG-20 operator UAT | parked | `1b14ffe` | PARK-24 pair | deferred |
| DIAG-21 Streamlit fixture audit | parked | `6b31678` | PARK-25 pair | deferred |
| Operator readiness + runtime hardening | parked | `91b9eba` | PARK-26 pair | maintenance only |
| Terminology helper/wiring mini-track | parked | `e398a75` | term helper/wiring pair | deferred (default-off, fail-closed) |
| License-gated private adapter track | **parked** | `b9b19ad` | PARK-02 pair (`medai-cka-term-license-gated-adapter-parked-2026-05-21`, `medai-final-parked-post-term-license-gate-2026-05-21`) | deferred until concrete operator license confirmations |
| Private terminology config boundary | **complete & verified** | `60f1114` | — | no action required |
| MeSH local helper/download boundary | blocked | `cedbbd3` | — | operator runs helper on Windows |
| Manual license verification gate | blocked | `376ca4e` | — | operator supplies concrete per-resource boolean confirmations |
| Cue expansion | **explicitly_not_recommended** | — | — | must not open |

## 5. Candidate next-phase ranking table

Rank = recommended execution order (1 = run first). Risk/cost/
reversibility columns are 1 (lowest) to 5 (highest). Operator-value
and product-value are 1 (lowest) to 5 (highest).

| Rank | ID | Candidate | Op. value | Product value | Impl. risk | Safety/privacy risk | Validation cost | Dependency risk | Expected time | Reversibility | Notes |
| ---: | :-: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | A | FREEZE-MAINTENANCE-ONLY | 4 | 4 | 1 | 1 | 1 | 1 | 1 | 5 | Hold the v1 freeze. Lowest-risk default after PARK-02. |
| 2 | F | V2-ARCHITECTURE-SPEC-01 | 4 | 5 | 2 | 1 | 2 | 2 | 3 | 5 | Reports-only spec; safest exit ramp for forward motion. |
| 3 | C | PACKAGING-DEPLOYMENT-NEXT-04 | 3 | 3 | 2 | 1 | 2 | 1 | 2 | 5 | Marginal value right after POLISH-03. |
| 4 | B | PRODUCT-UX-NEXT-01 | 4 | 4 | 3 | 2 | 3 | 2 | 3 | 4 | Reopens `app/main.py`; needs fresh UAT signals first. |
| 5 | E | REAL-CORPUS-VALIDATION-03 | 4 | 4 | 3 | **5** | 4 | 3 | 4 | 4 | Needs its own privacy-gated SPEC block first. |
| 6 | D | DATA-INFRA-NEXT-02 | 3 | 3 | 3 | 2 | 3 | 2 | 3 | 4 | DATA-RUNTIME-HARDEN-01 already closed immediate gaps. |
| 7 | G | TERMINOLOGY-LICENSE-WAIT | 3 | 3 | 1 | 1 | 1 | 1 | 1 | 5 | Passive holding pattern, effectively already in place via PARK-02. |
| 8 | H | MORE-UNKNOWN-DIAGNOSTICS | 2 | 2 | 2 | 2 | 3 | 2 | 3 | 4 | **Defer.** Track parked at ~99.99% closed. |
| 9 | I | CUE-EXPANSION | 2 | 2 | **5** | **5** | 5 | 5 | 5 | 2 | **Explicitly NOT recommended.** |

## 6. Top recommendation

**FREEZE-MAINTENANCE-ONLY (A).**

Rationale: after PARK-02, every expansion track is now parked, frozen,
or operator-side-blocked. No fresh failure signal is driving forward
motion. With no operator-UAT defect and no concrete operator license
confirmations, holding the v1 release as the durable shipped artifact
is the honest default. Forward motion, when desired, should open
through a reports-only architecture SPEC — never through cue
expansion, never through private adapter implementation under the
current license posture, and never through operator-UX changes
without a fresh operator-UAT signal.

## 7. Recommended next 3-block sequence

1. **FREEZE-MAINTENANCE-ONLY** — current posture; periodic re-runs of
   the five fixed health-check validations against the frozen v1
   release.
2. **V2-ARCHITECTURE-SPEC-01** — reports-only architecture spec
   (only when forward motion is desired). No runtime change. No
   license exposure. Produces a SPEC that any future v2 implementation
   block can anchor against.
3. **MEDAI-ROADMAP-07** — re-evaluate after V2-ARCHITECTURE-SPEC-01
   lands, or sooner if a fresh signal forces re-planning.

Alternative path (if packaging polish is preferred over a v2 spec):
replace V2-ARCHITECTURE-SPEC-01 with PACKAGING-DEPLOYMENT-NEXT-04
(low risk, marginal value).

## 8. Blocked / deferred work

Blocked (must not be reopened without an explicit new approved
planning block):

- Private adapter implementation
- Real private-store access
- Wiring private terminology output into `app/main.py` runtime beyond
  the parked helper/wiring mini-track
- DDI / diagnosis / treatment / medication inference driven by
  terminology
- Licensed terminology row reads
- Private license-acknowledgement file contents access
- External terminology API enablement for runtime
- **Cue-pack expansion (explicitly NOT recommended)**
- Public-report row dumps under any condition
- Auto-accept driven by terminology coding
- MeSH integration until both license confirmation AND completed
  operator-side download

Explicitly deferred:

- `MORE-UNKNOWN-DIAGNOSTICS`
- `CUE-EXPANSION`
- `PRIVATE-ADAPTER-IMPLEMENTATION`

## 9. Maintenance plan

Scope: maintain the frozen v1 local operator release; preserve every
park / freeze tag pair on origin; refrain from any runtime code
change.

Periodic validation set (unchanged from FREEZE):

- `python scripts/run_cka_final_mvp_release_validation.py`
- `python scripts/run_b07_term01_opt_in_integration_validation.py`
- `python scripts/run_medai_route_fix01_validation.py`
- `python scripts/run_medai_ui_ops_panel_validation.py`
- `python scripts/run_medai_ui_boot_fix_validation.py`

Expected conclusions:

| Validation | Expected conclusion |
| --- | --- |
| CKA MVP | `cka_mvp_release_package_ready` (693 tests; `external_api_used: false`) |
| B07 term01 | `cases_failed: 0`; `external_api_used: false` |
| ROUTE-FIX 01 | `medai_route_fix01_ready`; `passed: true` |
| UI ops panel | `medai_ui_ops_panel_ready` |
| UI boot fix | `medai_ui_boot_fix_startup_resilience_ready` |

Cadence: operator chooses (e.g. weekly, or before any planned demo).
Each run produces idempotent receipts under `reports/`. Any deviation
from the expected conclusions is treated as an escalation event.

Escalation conditions:

- Any of the five health-check validations prints a non-`_ready`
  conclusion two runs in a row.
- Any pre-existing tag appears moved, deleted, or repointed.
- Any private / terminology / PHI / key / secret / private-path
  artifact appears in `git status`, in a public report, or in any
  committed file.
- Any external API gets enabled by default in any deployment.
- Any operator-UAT signal surfaces a new defect against the frozen v1
  release.
- Any operator return supplies concrete per-resource license
  confirmations — that becomes the trigger for the deferred license-
  gated path.

## 10. Safety / privacy confirmation

- `runtime_behavior_changed` / `app_main_modified` / `helper_modified`
  / `streamlit_wiring_changed` / `launcher_files_modified` /
  `startup_preflight_modified` / `config_modified`: **false**.
- `extraction_behavior_changed` / `ocr_behavior_changed` /
  `classifier_behavior_changed` / `threshold_behavior_changed`:
  **false**.
- `cue_expansion_recommended` / `cue_expansion_performed`: **false**.
- `external_api_used_for_runtime` / `external_api_enabled`: **false**.
- `clinical_interpretation_performed` /
  `diagnosis_inference_performed` /
  `medication_inference_performed` /
  `ddi_behavior_changed` /
  `treatment_inference_performed`: **false**.
- `licensed_terminology_rows_read` /
  `licensed_terminology_rows_printed` /
  `licensed_terminology_rows_in_public_reports`: **false**.
- `license_ack_private_read`: **false**.
- `private_config_contents_read` / `private_config_staged`: **false**.
- `runtime_db_contents_opened` / `source_documents_opened` /
  `private_files_opened_for_content`: **false**.
- `raw_text_printed` / `raw_filenames_printed` /
  `private_paths_printed` / `secrets_printed`: **false**.
- `tags_created` / `tags_modified` / `prior_park_tags_touched`:
  **false**.

## 11. Why cue expansion remains NOT recommended

Reaffirmed. Across DIAG-13..21, ROADMAP-01..05, the FREEZE block,
the terminology chain (PLAN-01 through PARK-02), and now ROADMAP-06,
cue expansion has been ruled out as a primary lever. Adding cue packs
would re-open the classifier surface that the residual-Unknown track
explicitly stopped touching, expand the regression surface, create
new licensing / privacy responsibilities, and resolve no currently
failing operator outcome. `cue_expansion_recommended` stays
**false**.

## 12. Progress estimate

| Track | Before this block | After this block |
| --- | --- | --- |
| Whole MedAI project | ~96.0% done / ~4.0% remaining | ~96.0% done / ~4.0% remaining |

ROADMAP-06 is a planning audit and produces no runtime change. The
project percentage does not advance. Forward motion resumes when an
approved expansion block (recommended:
`MEDAI-V2-ARCHITECTURE-SPEC-01` reports-only spec) lands, or never
if the operator selects long-term freeze-maintenance posture.
