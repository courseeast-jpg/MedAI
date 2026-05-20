# MEDAI-ROADMAP-04 — Strategic Expansion Decision After Local Operator Release Freeze

Reports-only planning audit. No runtime change. No `app/main.py` /
launcher / preflight / config modification. No tags created. FREEZE
tag pair and PARK-20..23 tag pairs untouched. Cue expansion remains
explicitly **NOT** recommended.

## 1. Executive recommendation

The next MedAI posture should be:

**FREEZE-MAINTENANCE-ONLY (G)** — hold the v1 frozen local operator
release as the durable shipped artifact and resume forward motion only
through a tightly-scoped SPEC block.

Best operator-value-per-risk option immediately after a freeze: every
readiness criterion already passes, no parked track has any failure
signal pulling it back open, and no licensing / privacy surface needs
immediate attention.

When forward motion is desired, the cleanest exit ramp is to open a
reports-only SPEC block for the highest-medical-value next track —
terminology / coding integration — before any execution.

## 2. Current frozen release baseline

Pulled from `FREEZE-LOCAL-OPERATOR-RELEASE`, `PARK-26`,
`PACKAGING-DEPLOYMENT-POLISH-03`, and `OPERATOR-MANUAL-CONSOLIDATION-01`
(public-safe metadata only):

| Signal | Value |
| --- | :-: |
| `local_operator_release_frozen` | **true** |
| Freeze commit | `7ef8ffd` |
| Freeze tags resolve to freeze commit | **true** |
| `blocking_bugs_found` | False |
| `launch_readiness_ready` | True |
| `runtime_hardening_ready` | True |
| `release_handoff_ready` | True |
| `operator_workflow_ready` | True |
| `operator_manual_ready` | True |
| `technical_handoff_ready` | True |
| `packaging_discoverability_ready` | True |
| `cue_expansion_status` | **not_recommended** |
| Cumulative `runtime_behavior_changed` across the recent chain | False |
| Whole MedAI project | **~94.0%** done / ~6.0% remaining |

Branch `clinical-knowledge-architecture` at HEAD `1870cdb` (FREEZE
receipt refresh).

## 3. What is frozen

Captured by the freeze block:

- Local operator workflow (launch via `Start_MedAI_UI`, Run & Review,
  Advanced technical details read-only).
- Local-only environment defaults
  (`MEDAI_LOCAL_ONLY=1`, `MEDAI_ALLOW_EXTERNAL_API=0`,
  `MEDAI_REQUIRE_PII_SCRUB=1`, `MEDAI_PRIVACY_AUDIT=1`).
- Five fixed health-check validation commands
  (CKA MVP / B07 / ROUTE-FIX / UI ops / UI boot).
- Default-off env-gated metadata and read-only render plan for
  PDF text/layout quality (DIAG-17..19).
- Consolidated operator manual and technical handoff produced by
  `OPERATOR-MANUAL-CONSOLIDATION-01`.
- Public-report privacy boundary.
- All parked tracks (residual Unknown, PDF text/layout, DIAG-20 UAT,
  DIAG-21 fixture audit, operator readiness + runtime hardening).

Tag map intact:

| Pair | Commit |
| --- | --- |
| PARK-20 | `3e46461` |
| PARK-21 | `9f9e22d` |
| PARK-22 | `f4d3cc6` |
| PARK-23 | `748c32a` |
| FREEZE | `7ef8ffd` |

PARK-24 (`1b14ffe`), PARK-25 (`6b31678`), PARK-26 (`91b9eba`) remain
untagged.

## 4. Candidate next-phase ranking table

Rank score is the recommended execution order (1 = run first). All
risk / cost / reversibility columns are 1 (lowest) to 5 (highest).
Operator-value and product-value are 1 (lowest) to 5 (highest).

| Rank | ID | Candidate | Op. value | Product value | Impl. risk | Safety/privacy risk | Validation cost | Dependency risk | Expected time | Reversibility | Notes |
| ---: | :-: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | G | FREEZE-MAINTENANCE-ONLY | 4 | 4 | 1 | 1 | 1 | 1 | 1 | 5 | Hold the v1 freeze. Lowest-risk default after release. |
| 2 | B | CKA-TERM-INTEGRATION-NEXT-01 | 4 | 5 | 4 | 4 | 4 | 4 | 4 | 3 | Highest medical value. Requires a SPEC block first (license + privacy gates). |
| 3 | A | PRODUCT-UX-NEXT-01 | 4 | 4 | 3 | 2 | 3 | 2 | 3 | 4 | Reopens `app/main.py`; needs fresh UAT signals first. |
| 4 | F | PACKAGING-DEPLOYMENT-NEXT-04 | 3 | 3 | 2 | 1 | 2 | 1 | 2 | 5 | Marginal value right after `PACKAGING-DEPLOYMENT-POLISH-03`. |
| 5 | D | REAL-CORPUS-VALIDATION-03 | 4 | 4 | 3 | **5** | 4 | 3 | 4 | 4 | Needs its own privacy-gated SPEC block first. |
| 6 | E | DATA-INFRA-NEXT-02 | 3 | 3 | 3 | 2 | 3 | 2 | 3 | 4 | `DATA-RUNTIME-HARDEN-01` already closed immediate gaps. |
| 7 | C | V2-ARCHITECTURE-BLOCK-01 | 3 | 5 | **5** | 3 | 5 | 5 | 5 | 2 | Premature without a `V2-ARCHITECTURE-SPEC` block. |
| 8 | H | MORE-UNKNOWN-DIAGNOSTICS | 2 | 2 | 2 | 2 | 3 | 2 | 3 | 4 | **Defer.** Track parked at ~99.99% closed. |
| 9 | I | CUE-EXPANSION | 2 | 2 | **5** | **5** | 5 | 5 | 5 | 2 | **Explicitly NOT recommended.** |

## 5. Top recommended next phase

**FREEZE-MAINTENANCE-ONLY (G).**

Scope outline (for the next prompt to anchor against, not a commitment
of work in this block):

- No runtime change.
- No new tags, no tag movement.
- Periodic re-runs of the five fixed health-check validation commands
  (CKA MVP / B07 / ROUTE-FIX / UI ops / UI boot) at the cadence the
  operator requires; each re-run produces public-safe receipts under
  `reports/`.
- Re-run the public-report privacy checker over `reports/` on a
  schedule the maintainer chooses.
- Treat any deviation in the five health-check `_ready` conclusions
  or in PARK / FREEZE tag pointers as an escalation event.

If and when forward motion is desired, open a SPEC block first — see
Section 6.

## 6. Recommended next 3-block sequence

1. **FREEZE-MAINTENANCE-ONLY** — hold the v1 freeze (current
   posture).
2. **CKA-TERM-INTEGRATION-PLAN-01** — a reports-only SPEC block that
   produces:
   - explicit license-class table for every terminology / coding
     source the project plans to integrate,
   - privacy gates (no licensed rows read into reports; aggregate-only
     outputs),
   - license-aware test plan,
   - rollback path for each integration step.
   This SPEC block must not touch licensed terminology rows or runtime
   code.
3. **CKA-TERM-INTEGRATION-NEXT-01 (B)** — only after the SPEC is
   approved, execute the terminology / coding integration under the
   planned constraints.

Alternative path (if v2 architecture is preferred over terminology
integration): replace blocks 2 and 3 with
`V2-ARCHITECTURE-SPEC-01` (reports-only architecture spec) and the
first safe sub-block of v2 work under block-phase rules. Both
alternatives keep the freeze invariants intact.

## 7. Explicitly deferred work

- **MORE-UNKNOWN-DIAGNOSTICS (H)** — deferred. Residual-Unknown
  reduction track is parked at ~99.99% closed across DIAG-13A..21 with
  parking tags PARK-20..23 and untagged snapshots PARK-24/25. Reopen
  only on a fresh operator-UAT signal.
- **CUE-EXPANSION (I)** — explicitly **NOT** recommended. Standing
  project posture across DIAG-13..21 and ROADMAP-01..03.
- **Broad v2 architecture** — deferred. Must open with a dedicated
  SPEC block first.
- **Real-corpus validation execution (D)** — deferred. Must open with
  a privacy-gated SPEC block first.
- **`app/main.py` UI surface changes (A)** — deferred. Needs fresh
  operator-UAT signals first.

## 8. Safety / privacy constraints

- No source documents opened.
- No raw OCR text, raw document text, raw filenames, private paths,
  PHI, secrets, runtime DB rows, backups, bundles, keys, or licensed
  terminology data read.
- No runtime DB contents inspected (only public-safe report metadata
  was read).
- No `app/main.py` / launcher / preflight / config modification.
- No tags created, moved, or deleted.
- No external APIs called or enabled.
- FREEZE tag pair and PARK-20..23 tag pairs unchanged on origin.

## 9. Why cue expansion remains NOT recommended

Across DIAG-13..21 and ROADMAP-01..03 and the FREEZE block, cue
expansion has been ruled out as a primary lever. Adding cue packs
would reopen the classifier surface that the residual-Unknown track
explicitly stopped touching, expand the regression surface, create
new privacy / licensing responsibilities, and resolve no currently
failing operator outcome. ROADMAP-04 reaffirms:
`cue_expansion_status = not_recommended`.

## 10. Why residual Unknown diagnostics remain parked

The 21-record residual Unknown surface (11 Sub-track A + 10 Sub-track
B) has been driven through DIAG-13A..21 with default-off metadata,
default-off render plan, env-gated Streamlit wiring, env-on aggregate
UAT, and a Streamlit fixture audit. PARK-26 confirms
`blocking_bugs_found=false` and `hardening_change_needed=false`; the
FREEZE block then froze the result. Reopening DIAG-22+ now would
re-touch a parked surface without changing any observable operator
outcome. Reopen only if a future operator UAT block records a fresh
Unknown-related failure.

## 11. Suggested Claude Code prompt title for the next block

If staying at maintenance posture:
`MEDAI-FREEZE-MAINTENANCE-ONLY — Maintain Frozen Local Operator Release`

If opening forward motion (recommended SPEC track):
`MEDAI-CKA-TERM-INTEGRATION-PLAN-01 — Reports-Only Spec for Terminology / Coding Integration After v1 Freeze`

## 12. Progress estimate

| Track | Before ROADMAP-04 | After ROADMAP-04 |
| --- | --- | --- |
| Whole MedAI project | ~94.0% done / ~6.0% remaining | ~94.0% done / ~6.0% remaining |

ROADMAP-04 is a planning audit and produces no runtime change, so the
project percentage does not advance. Forward motion resumes when an
approved expansion block (e.g. `CKA-TERM-INTEGRATION-PLAN-01`) lands.
