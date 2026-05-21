# MEDAI-V2-ARCHITECTURE-SPEC-01 — Reports-Only V2 Architecture Planning Spec

Reports-only architecture SPEC. Defines the v2 direction without
implementing anything. Preserves the frozen v1 local operator release.
No runtime change. No tags created. PARK-20..23 / PARK-24..26 / FREEZE
/ TERM helper-wiring PARK-01 / license-gate PARK-02 tag pairs all
remain intact. Cue expansion remains explicitly **NOT** recommended.

## 1. Executive summary

`MEDAI-V2-ARCHITECTURE-SPEC-01` defines:

- 10 architecture goals (preserve v1 freeze; local-only; HITL; no
  silent auto-accept; no external API default; private/licensed
  terminology gates; no cue expansion; layered separation; per-track
  parking; public-safe reports only).
- 10 architecture layers (A–J: ingestion / extraction / classification
  / structured medical extraction / clinical knowledge & terminology /
  safety & decision / operator UI / runtime data & persistence /
  observability & validation / packaging & deployment).
- 9 workstreams (`V2-FOUNDATION-SPEC-02`, `V2-RUNTIME-CONTRACTS-01`,
  `V2-VALIDATION-HARNESS-01`, `V2-UI-SHELL-SPEC-01`,
  `V2-DATA-INFRA-SPEC-01`, `V2-EXTRACTION-SPEC-01`,
  `V2-TERMINOLOGY-WAIT-GATE`, `V2-PACKAGING-SPEC-01`,
  `V2-ROADMAP-02`).
- Block-phase split rules and combination guidance.
- A validation matrix that carries forward the v1 five-validation
  health-check set unchanged and adds per-block privacy + focused-test
  requirements.
- A parking strategy mirroring the established pattern.
- An explicit non-goals list including cue expansion, private adapter
  implementation, licensed row reads, external terminology APIs,
  autonomous clinical inference, DDI behavior changes, real-corpus
  validation without a privacy-gated SPEC, runtime DB migrations
  without a rollback plan, and production deployment automation.

Top recommended next block: **`V2-FOUNDATION-SPEC-02`**. Recommended
next 3-block sequence: `V2-FOUNDATION-SPEC-02` →
`V2-RUNTIME-CONTRACTS-01` → `V2-VALIDATION-HARNESS-01`.

## 2. Why V2-ARCHITECTURE-SPEC-01 exists after ROADMAP-06

`MEDAI-ROADMAP-06` (commit `5ea7bd6`) selected
**`FREEZE-MAINTENANCE-ONLY`** as the honest default after PARK-02 and
allowed `V2-ARCHITECTURE-SPEC-01` as the clean forward-motion exit
ramp. This SPEC executes that exit ramp safely: reports-only, no
runtime change, no license exposure, durable architecture record.

## 3. Frozen v1 baseline

| Signal | Value |
| --- | :-: |
| Branch | `clinical-knowledge-architecture` |
| HEAD before this block | `1bf86dd` |
| Freeze commit | `7ef8ffd` |
| `local_operator_release_frozen` | true |
| `v1_release_preserved` | true |
| `cumulative_runtime_behavior_changed_across_recent_chain` | false |
| `cue_expansion_recommended` | false |
| `external_api_used` / `external_api_enabled` | false / false |
| Whole MedAI project | **~96.1%** done / ~3.9% remaining |

## 4. Current parked / blocked tracks (carried forward)

| Track | Status | Commit |
| --- | --- | --- |
| Local operator release | **frozen** | `7ef8ffd` |
| Residual-Unknown reduction | parked | `3e46461` |
| Text-layer eval spec | parked | `9f9e22d` |
| PDF text/layout default-off | parked | `f4d3cc6` |
| PDF text/layout Streamlit wiring | parked | `748c32a` |
| DIAG-20 operator UAT | parked | `1b14ffe` |
| DIAG-21 fixture audit | parked | `6b31678` |
| Operator readiness + runtime hardening | parked | `91b9eba` |
| Terminology helper/wiring mini-track | parked | `e398a75` |
| License-gated private adapter | parked | `b9b19ad` |
| Private terminology config boundary | complete & verified | `60f1114` |
| MeSH local helper / download | blocked (operator-side) | `cedbbd3` |
| Manual license verification gate | blocked (operator-side) | `376ca4e` |
| Cue expansion | **explicitly_not_recommended** | — |

## 5. V2 architecture goals

1. Preserve the frozen v1 local operator release (`7ef8ffd`, FREEZE
   tag pair).
2. Keep default local-only posture (`MEDAI_LOCAL_ONLY=1`,
   `MEDAI_ALLOW_EXTERNAL_API=0`, `MEDAI_REQUIRE_PII_SCRUB=1`,
   `MEDAI_PRIVACY_AUDIT=1`).
3. Preserve the human-in-the-loop review boundary for every emitted
   result.
4. Preserve no-silent-auto-accept: `accepted_count` /
   `auto_accept_allowed_count` are workflow tallies, never clinical
   acceptance counts.
5. Preserve no-external-API-by-default for runtime.
6. Preserve private / licensed terminology gates established by PARK-02
   and the prior TERM chain.
7. Keep cue expansion disallowed unless a future approved SPEC
   reverses it explicitly.
8. Separate runtime pipeline, operator UI, terminology, validation,
   storage, observability, and deployment into independently gated
   workstreams.
9. Require a parking snapshot (commit + two annotated tags) after each
   high-risk track.
10. Require public-safe reports only; aggregate-only outputs.

## 6. V2 architecture layers

| ID | Layer | Concerns | This-SPEC scope |
| :-: | --- | --- | --- |
| A | Ingestion and source handling | file intake, local-only, privacy gates, source quarantine, no public raw text | design only |
| B | Extraction / OCR | PDF text quality, OCR routing, adapters, confidence / fallback isolation | design only |
| C | Document classification | classifier contracts, confidence bands, review-bound defaults, unknown handling | design only — no cue expansion |
| D | Structured medical extraction | labs / medications / findings | future work only; no inference by default |
| E | Clinical knowledge / terminology | parked helper/wiring baseline; parked license-gated adapter | design only — private adapter blocked |
| F | Safety and decision | refusal rules, no autonomous diagnosis/treatment, DDI parked | design only |
| G | Operator UI | Run & Review, Advanced technical details, local-only status | design only |
| H | Runtime data and persistence | DB safety, no row reads, migration gates, rollback expectations | design only |
| I | Observability and validation | v1 five-validation health check carried forward unchanged | design only |
| J | Packaging / deployment | local launcher, runbook, freeze / parking rules | design only |

## 7. V2 workstream plan

| ID | Risk | Allowed mode | Parking requirement | Key exclusions |
| --- | :-: | --- | --- | --- |
| `V2-FOUNDATION-SPEC-02` | low | reports-only | none (SPEC only) | runtime code, UI, terminology / license / adapter implementation, cue expansion |
| `V2-RUNTIME-CONTRACTS-01` | low-moderate | reports-only + optional typing-only module | tag once it lands | concrete adapter impl, `app/main.py` change, Streamlit wiring |
| `V2-VALIDATION-HARNESS-01` | low | reports-only + focused tests | none (validation is durable) | runtime behavior changes, modification of v1 validation scripts |
| `V2-UI-SHELL-SPEC-01` | low-moderate | reports-only | tag once it lands | `app/main.py` change, Streamlit code in this SPEC |
| `V2-DATA-INFRA-SPEC-01` | moderate | reports-only | tag once it lands | DB schema change, DB row read, private path printed |
| `V2-EXTRACTION-SPEC-01` | moderate | reports-only | tag once it lands | OCR routing change, extraction behavior change, cue packs |
| `V2-TERMINOLOGY-WAIT-GATE` | minimal | passive wait | no parking tag | private adapter impl, real private-store access, MeSH integration, licensed row reads |
| `V2-PACKAGING-SPEC-01` | low | reports-only | tag once it lands | launcher behavior change in SPEC, deployment automation |
| `V2-ROADMAP-02` | minimal | reports-only | no parking tag | any implementation work |

Per-workstream scope, stop conditions, and validation requirements are
in the JSON under `v2_workstream_plan`.

## 8. Block-phase split guidance

**Must split** if a block would touch any of: OCR routing, extraction
behavior, thresholds / scoring, privacy gates, terminology license
gates, clinical decision logic, DDI behavior, runtime DB writes,
auto-accept behavior, or default-on changes to any env-gated feature.

**May combine** when all hold: all sub-tasks are reports-only; all
sub-tasks share a single architecture layer; no sub-task changes
runtime behavior; no sub-task touches a parked / frozen tag; no
sub-task reads licensed rows / private contents; total combined
report set passes the public-report privacy check.

**Never combine**: unrelated architecture layers, reports-only +
runtime-changing work, license-gated terminology work with anything
else, cue expansion with anything (cue expansion remains explicitly
**NOT** recommended), or private-adapter implementation with any other
work.

## 9. Validation matrix

V1 five health-check validations carry forward unchanged:

| Command | Expected conclusion |
| --- | --- |
| `python scripts/run_cka_final_mvp_release_validation.py` | `cka_mvp_release_package_ready` (693 tests; `external_api_used: false`) |
| `python scripts/run_b07_term01_opt_in_integration_validation.py` | `cases_failed: 0`; `external_api_used: false` |
| `python scripts/run_medai_route_fix01_validation.py` | `medai_route_fix01_ready`; `passed: true` |
| `python scripts/run_medai_ui_ops_panel_validation.py` | `medai_ui_ops_panel_ready` |
| `python scripts/run_medai_ui_boot_fix_validation.py` | `medai_ui_boot_fix_startup_resilience_ready` |

Per-v2-block required additions:

- Public-report privacy check on every new public report
  (`clinical_knowledge.privacy.check_public_report_payload`).
- Staged safety check (no terminology data, no private config, no
  `LICENSE_ACK_PRIVATE`, no MeSH files, no private paths in any
  commit).
- Focused pytest module per implementation block (mirroring DIAG-19/20
  / terminology-helper test patterns).

Every v2 block must emit the standard invariant flags as **false**:
`runtime_behavior_changed`, `app_main_modified`, `helper_modified`,
`streamlit_wiring_changed`, `launcher_files_modified`,
`startup_preflight_modified`, `config_modified`,
`extraction_behavior_changed`, `ocr_behavior_changed`,
`classifier_behavior_changed`, `threshold_behavior_changed`,
`cue_expansion_recommended`, `cue_expansion_performed`,
`external_api_used`, `external_api_enabled`,
`clinical_interpretation_performed`, `diagnosis_inference_performed`,
`medication_inference_performed`, `ddi_behavior_changed`,
`treatment_inference_performed`, `licensed_terminology_rows_read`,
`license_ack_private_read`, `raw_text_printed`,
`raw_filenames_printed`, `private_paths_printed`, `secrets_printed`,
`tags_created` (except parking blocks), `tags_modified`.

## 10. Parking strategy

- Every high-risk track ends with a parking snapshot: one named
  commit + two annotated tags (`{track}-ready-{date}` and
  `medai-final-parked-post-{track}-{date}`).
- Parking tags are pushed via github-direct (the local proxy
  historically 403s on `refs/tags/*` writes).
- Never use `git push --tags`. Never use `--force` /
  `--force-with-lease`. Tag pushes are explicit by name.
- Existing parking tags (PARK-20..26, FREEZE, term helper/wiring
  PARK-01, license-gate PARK-02) remain unchanged across every
  subsequent block.
- A parking block is reports-only by definition: it records the
  matrix, blocks the implementation, and adds the two annotated
  tags.
- Reopening a parked track requires a fresh SPEC block first;
  reopening must not move or repoint existing parking tags.

## 11. Non-goals / deferred items

- Private adapter implementation
- Real private-store access
- Licensed terminology row reads in any v2 block
- External terminology APIs (UMLS REST, NLM web services, SNOMED
  Cloud, RxNorm cloud, MeSH cloud) at runtime
- Cue expansion
- Autonomous diagnosis or treatment
- Medication / dose inference
- DDI behavior changes driven by terminology coding
- Real-corpus validation until a privacy-gated SPEC is separately
  approved
- `MORE-UNKNOWN-DIAGNOSTICS`
- Runtime DB migrations without a SPEC + rollback plan
- Production deployment automation
- Public-report row dumps under any condition
- Default-on changes to any env-gated feature

## 12. Safety / privacy constraints

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

## 13. Why cue expansion remains NOT recommended

Reaffirmed across DIAG-13..21, ROADMAP-01..06, the FREEZE block, the
terminology chain (PLAN-01 through PARK-02), and now V2-ARCHITECTURE-
SPEC-01. Adding cue packs would re-open the classifier surface that
the residual-Unknown track explicitly stopped touching, expand the
regression surface, create new licensing / privacy responsibilities,
and resolve no currently failing operator outcome.
`cue_expansion_recommended` stays **false** and remains a v2 non-goal.

## 14. Recommended next 3-block sequence

1. **`V2-FOUNDATION-SPEC-02`** — reports-only foundation SPEC that
   restates the cumulative invariants and the canonical termination
   conditions for any v2 block.
2. **`V2-RUNTIME-CONTRACTS-01`** — reports-only contracts (Python
   `typing.Protocol` / `dataclass` only) for v2 runtime interfaces.
3. **`V2-VALIDATION-HARNESS-01`** — reports-only v2 validation matrix
   SPEC plus a focused pytest harness cataloguing the existing v1
   five-validation set with reports-only stubs for v2 contract
   conformance.

Each block is reports-only, default-off-by-design, aggregate-only,
review-bound, and explicitly excludes runtime / UI / terminology /
license / adapter / cue-expansion work.

## 15. Progress estimate

| Track | Before this block | After this block |
| --- | --- | --- |
| Whole MedAI project | ~96.0% done / ~4.0% remaining | ~96.1% done / ~3.9% remaining |

V2-ARCHITECTURE-SPEC-01 is a reports-only architecture SPEC. The
small bump reflects the durable architecture plan; no runtime change.
