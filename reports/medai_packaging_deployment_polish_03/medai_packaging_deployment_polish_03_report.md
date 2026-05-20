# MEDAI-PACKAGING-DEPLOYMENT-POLISH-03 — Launcher, Install, and First-Run UX Polish

Low-risk docs polish. Three top-level docs gained a short pointer block
each so the consolidated operator manual and technical handoff (from
`OPERATOR-MANUAL-CONSOLIDATION-01`) become discoverable from the docs
operators read first. No launcher / preflight / config / runtime
change. No tags created. PARK-20..26 untouched. Cue expansion remains
explicitly **NOT** recommended.

## 1. Why PACKAGING-DEPLOYMENT-POLISH-03 exists

`ROADMAP-03` selected `OPERATOR-MANUAL-CONSOLIDATION-01` as the next
phase and named `PACKAGING-DEPLOYMENT-POLISH-03` as the immediate
follow-up. The consolidation block produced two durable documents:

- `reports/medai_operator_manual_consolidation_01/MEDAI_LOCAL_OPERATOR_MANUAL.md`
- `reports/medai_operator_manual_consolidation_01/MEDAI_TECHNICAL_HANDOFF.md`

But before this block, none of the three top-level docs an operator or
maintainer hits first (`README.md`, `RELEASE_QUICKSTART_LOCAL_ONLY.md`,
`RELEASE_OPERATOR_GUIDE.md`) referenced them. The new consolidated
docs were effectively hidden from the natural first-run path.

`PACKAGING-DEPLOYMENT-POLISH-03` closes that gap with the smallest
possible intervention: a single ~8–12-line pointer block at the top of
each of the three docs.

## 2. Current launch / handoff surface

| Surface | File | Status |
| --- | --- | --- |
| Operator install / setup | `README.md` | Updated: new "Latest operator documentation" section pointing to both consolidated docs. |
| Operator quickstart | `RELEASE_QUICKSTART_LOCAL_ONLY.md` | Updated: new "Latest operator manual" section pointing to both consolidated docs. |
| Operator guide (legacy HITL release) | `RELEASE_OPERATOR_GUIDE.md` | Updated: new "Latest consolidated operator documentation" section. |
| Limitations / safety | `RELEASE_LIMITATIONS_AND_SAFETY.md` | Unchanged. |
| Architecture | `ARCHITECTURE.md` | Unchanged. |
| Launchers (4 files) | `Start_MedAI_UI.bat`, `Start_MedAI_UI_Silent.vbs`, `Start_MedAI_Test_UI.bat`, `Start_MedAI_UI_Encrypted.bat` | **Unchanged.** Already emit adequate first-run text (`Starting MedAI...`, `Local-only mode ON`, `External APIs disabled by privacy gate`, `Browser opening at localhost:8501`). |
| Startup preflight | `app/startup_preflight.py` | Unchanged. |
| Config | `app/config.py` | Unchanged. |
| Runtime UI | `app/main.py` | Unchanged. |
| Parking tags | `PARK-20..23` tag pairs | Unchanged on origin. |

## 3. Changes made

### 3a. `README.md` — added section after intro, before "Prerequisites"

A new `## Latest operator documentation` block listing the two
consolidated docs and explaining that the rest of the README + legacy
release docs remain authoritative for their original topics.

### 3b. `RELEASE_QUICKSTART_LOCAL_ONLY.md` — added section at top, before "1. Start the UI"

A new `## Latest operator manual` block pointing to both consolidated
docs and telling the operator to use this quickstart for the minimal
first-run sequence and the consolidated operator manual for full
guidance / health checks / recovery / escalation.

### 3c. `RELEASE_OPERATOR_GUIDE.md` — added section at top, before the legacy "What do I" section

A new `## Latest consolidated operator documentation` block pointing to
both consolidated docs. The legacy guide remains authoritative for the
OCR/Layout HITL release frame described below; the consolidated
operator manual is the navigation index for current local operator
use.

### 3d. What was NOT changed

- `RELEASE_LIMITATIONS_AND_SAFETY.md` — not a first-run doc; left
  authoritative for safety boundaries.
- `ARCHITECTURE.md` — not a first-run doc; left authoritative for
  architecture.
- The four shipped launchers — already adequate first-run text; the
  task's "default posture: prefer docs/report-only" governs.
- `app/main.py`, `app/startup_preflight.py`, `app/config.py` — out of
  scope; parked operator-readiness surface.
- Pre-existing PERSON / MEDICAL_FILENAME findings inside the legacy
  operator guide doc — predate this block and were left unchanged.
  They are docs-only triggers (older content), not public-report
  leaks. Cleaning them up would be a follow-up candidate.

## 4. First-run operator path

After this polish, the first-run operator path is:

1. Operator opens `README.md` (or directly `RELEASE_QUICKSTART_LOCAL_ONLY.md`).
2. The top of the doc immediately points to
   `reports/medai_operator_manual_consolidation_01/MEDAI_LOCAL_OPERATOR_MANUAL.md`.
3. Operator reads the consolidated manual end-to-end (11 sections).
4. Operator launches with `Start_MedAI_UI.bat` (or
   `Start_MedAI_UI_Silent.vbs`).
5. Operator verifies health via the five fixed validation commands the
   consolidated manual lists in §3.

A maintainer's path is the same except they branch into
`MEDAI_TECHNICAL_HANDOFF.md` after step 2.

## 5. Manual and handoff discoverability

| Doc | Pointer present | Source line range |
| --- | :-: | --- |
| `README.md` | ✓ | New "Latest operator documentation" section. |
| `RELEASE_QUICKSTART_LOCAL_ONLY.md` | ✓ | New "Latest operator manual" section. |
| `RELEASE_OPERATOR_GUIDE.md` | ✓ | New "Latest consolidated operator documentation" section. |

A focused test in
`tests/test_medai_packaging_deployment_polish_03.py` asserts each
pointer is present and that the runtime / launcher / preflight / config
surface is untouched.

## 6. Local-only posture

The shipped local-only environment defaults are unchanged:

| Env var | Default | Source |
| --- | :-: | --- |
| `MEDAI_LOCAL_ONLY` | `1` | `PACKAGING-LAUNCHER-HARDEN-01` |
| `MEDAI_ALLOW_EXTERNAL_API` | `0` | `PACKAGING-LAUNCHER-HARDEN-01` |
| `MEDAI_REQUIRE_PII_SCRUB` | `1` | `PACKAGING-LAUNCHER-HARDEN-01` |
| `MEDAI_PRIVACY_AUDIT` | `1` | `PACKAGING-LAUNCHER-HARDEN-01` |

`local_only_posture_preserved`: **true**. `external_api_used`: **false**.
`external_api_enabled`: **false**.

## 7. Validation evidence

| Validation | Result |
| --- | --- |
| Public-report privacy checks (3 PACKAGING-DEPLOYMENT-POLISH-03 reports) | PASS |
| Focused polish tests (`tests/test_medai_packaging_deployment_polish_03.py`) | PASS |
| Final CKA MVP validation | PASS (`cka_mvp_release_package_ready`; 693 tests; `external_api_used: false`) |
| B07 term01 opt-in integration | PASS (`cases_failed: 0`, `external_api_used: false`) |
| ROUTE-FIX 01 | PASS (`medai_route_fix01_ready`, `passed: true`) |
| UI ops panel | PASS (`medai_ui_ops_panel_ready`) |
| UI boot fix | PASS (`medai_ui_boot_fix_startup_resilience_ready`) |
| Staged safety check | PASS — only PACKAGING-DEPLOYMENT-POLISH-03 scoped files + the 3 modified top-level docs staged into the implementation commit; validation receipt churn isolated to a separate receipt-refresh commit. |
| Full pytest | Not run. Reports/docs-only block; no runtime change. Known Streamlit-import sandbox limitation persists but is not a regression. |

## 8. Safety / privacy confirmation

- `runtime_behavior_changed`: false
- `app_main_modified`: false
- `launcher_files_modified`: false
- `startup_preflight_modified`: false
- `config_modified`: false
- `extraction_behavior_changed` / `ocr_behavior_changed`
  / `classifier_behavior_changed` / `threshold_behavior_changed`: false
- `cue_expansion_recommended` / `cue_expansion_performed`: false
- `external_api_used` / `external_api_enabled`: false
- `source_documents_opened` / `private_files_opened`
  / `runtime_db_contents_opened`
  / `licensed_terminology_rows_read`: false
- `raw_text_printed` / `raw_filenames_printed`
  / `private_paths_printed` / `secrets_printed`: false
- `clinical_value_parsing_performed`
  / `clinical_interpretation_performed`
  / `diagnosis_inference_performed`
  / `medication_inference_performed`
  / `ddi_inference_performed`
  / `treatment_inference_performed`
  / `abbreviation_expansion_performed`: false
- `tags_created` / `tags_modified` / `prior_park_tags_touched`: false
- `local_only_posture_preserved`: **true**

## 9. Remaining packaging risks

1. Legacy `RELEASE_QUICKSTART_LOCAL_ONLY.md` still references API-key
   setup (`ANTHROPIC_API_KEY` / `GEMINI_API_KEY`) from an earlier
   project phase. The shipped default is local-only (`MEDAI_LOCAL_ONLY=1`,
   `MEDAI_ALLOW_EXTERNAL_API=0`); the legacy text is informational and
   does not enable external APIs. Cleaning it up is a follow-up
   candidate for `MEDAI-FREEZE-LOCAL-OPERATOR-RELEASE` or a later
   docs-only block.
2. The legacy operator guide doc carries pre-existing PERSON /
   MEDICAL_FILENAME privacy-scanner findings from older content. They
   are docs-only triggers, not public-report leaks. Pointer-block
   additions in this block were verified clean; the pre-existing
   findings were left unchanged.
3. Launchers were intentionally not modified. If operator UAT later
   surfaces friction at the launch step, a focused, audited launcher
   block can be opened then.

## 10. Recommended next step

**`MEDAI-FREEZE-LOCAL-OPERATOR-RELEASE`** — final release snapshot of
the local operator artifact, after which `MEDAI-ROADMAP-04` should
re-evaluate the next strategic phase before opening any v2 /
terminology / real-corpus expansion block.

## 11. Cue expansion remains NOT recommended

Reaffirmed across every block in this chain: cue expansion would
re-open the classifier surface, expand the regression surface, create
new privacy / licensing responsibilities, and resolve no currently
failing operator outcome. The standing project posture is
**`cue_expansion_recommended: false`**.

## Progress estimate

| Track | Before | After |
| --- | --- | --- |
| Whole MedAI project | ~93.2% done / ~6.8% remaining | ~93.5% done / ~6.5% remaining |
