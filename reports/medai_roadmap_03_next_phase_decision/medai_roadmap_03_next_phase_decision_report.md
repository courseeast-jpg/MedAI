# MEDAI-ROADMAP-03 — Next Phase Decision After Operator Runtime Readiness

Reports-only planning audit. No runtime change. No app/main.py / launcher
/ preflight / config modification. No tags created. PARK-20..26 tags
untouched. Cue expansion remains explicitly NOT recommended.

## 1. Executive recommendation

The next MedAI phase should be:

**OPERATOR-MANUAL-CONSOLIDATION-01** — consolidate the operator manual
and technical handoff into a single coherent document set.

It is the best operator-value-per-risk option at the current state:
every input already exists as a public-safe report; the work is pure
docs/text; it cannot regress runtime behavior; it cannot leak private
data; and it lays the documentation foundation that PACKAGING-
DEPLOYMENT-POLISH-03 and FREEZE-LOCAL-OPERATOR-RELEASE will both
reference.

If stability matters more than consolidation, the alternative
acceptable top recommendation is **FREEZE-LOCAL-OPERATOR-RELEASE (H)**
directly. Every freeze criterion already passes; the trade-off is
losing the consolidation gain.

## 2. Current readiness baseline

Pulled from `PARK-26` and `DATA-RUNTIME-HARDEN-01` reports (public-safe
metadata only):

| Signal | Value |
| --- | :-: |
| `blocking_bugs_found` | False |
| `launch_readiness_ready` | True |
| `runtime_hardening_ready` | True |
| `release_handoff_ready` | True |
| `operator_workflow_ready` | True |
| `hardening_change_needed` | False |
| `cue_expansion_recommended` | False |
| `runtime_behavior_changed` (cumulative across recent blocks) | False |
| `local_operator_release_ready` | **True** |
| Whole MedAI project | **~93.0%** done / ~7.0% remaining |

Branch `clinical-knowledge-architecture` at HEAD `91b9eba` (PARK-26).

## 3. Tracks now parked

| Track | Status | Latest parking tag pair |
| --- | --- | --- |
| Residual-Unknown reduction | **parked** | PARK-20 → `3e46461`, PARK-21 → `9f9e22d` |
| PDF text/layout quality | **parked** | PARK-22 → `f4d3cc6`, PARK-23 → `748c32a` |
| DIAG-20 operator UAT | **parked** | PARK-24 → `1b14ffe` |
| DIAG-21 Streamlit fixture audit | **parked** | PARK-25 → `6b31678` |
| Operator readiness + runtime hardening | **parked** | PARK-26 → `91b9eba` |

All tag pairs remain on origin. ROADMAP-03 must not touch any of them.

## 4. Candidate next-phase ranking table

Rank score is the recommended execution order (1 = run first). All
risk/cost columns are 1 (lowest) to 5 (highest). Operator-value and
product-value are 1 (lowest) to 5 (highest).

| Rank | ID | Candidate | Op. value | Product value | Impl. risk | Safety/privacy risk | Validation cost | Dependency risk | Expected time | Notes |
| ---: | :-: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | G | OPERATOR-MANUAL-CONSOLIDATION-01 | 5 | 4 | 1 | 1 | 1 | 1 | 2 | Pure docs work over existing public-safe reports. |
| 2 | E | PACKAGING-DEPLOYMENT-POLISH-03 | 3 | 3 | 2 | 1 | 2 | 1 | 2 | Closes the launcher/install/UX loop after G lands. |
| 3 | H | FREEZE-LOCAL-OPERATOR-RELEASE | 4 | 4 | 1 | 1 | 2 | 1 | 1 | Final snapshot after G + E; criteria already met. |
| 4 | A | PRODUCT-UX-NEXT-01 | 4 | 4 | 3 | 2 | 3 | 2 | 3 | Reopens `app/main.py`; needs fresh UAT signals first. |
| 5 | D | REAL-CORPUS-VALIDATION-03 | 4 | 4 | 3 | **5** | 4 | 3 | 4 | Needs its own privacy-gated SPEC block first. |
| 6 | F | DATA-INFRA-NEXT-02 | 3 | 3 | 3 | 2 | 3 | 2 | 3 | DATA-RUNTIME-HARDEN-01 already closed immediate gaps. |
| 7 | B | CKA-TERM-INTEGRATION-NEXT-01 | 4 | 5 | 4 | 4 | 4 | 4 | 4 | Crosses licensing/privacy lines; needs a planning block first. |
| 8 | C | V2-ARCHITECTURE-BLOCK-01 | 3 | 5 | **5** | 3 | 5 | 5 | 5 | Premature without a v2 trigger; freeze v1 first. |
| 9 | I | MORE-UNKNOWN-DIAGNOSTICS | 2 | 2 | 2 | 2 | 3 | 2 | 3 | **Defer.** Track is parked at ~99.99% closed. |
| 10 | J | CUE-EXPANSION | 2 | 2 | **5** | **5** | 5 | 5 | 5 | **Explicitly NOT recommended.** |

## 5. Top recommended next phase

**OPERATOR-MANUAL-CONSOLIDATION-01 (G).**

Scope outline (for the next prompt to anchor against, not a commitment
of work in this block):

- Consolidate `RELEASE-HANDOFF-01`, `REAL-WORLD-OPERATOR-UAT-02`,
  `UI-USABILITY-POLISH-01`, `DATA-RUNTIME-HARDEN-01` and `PARK-26`
  guidance into a single operator manual (sectioned: launch, validate,
  monitor, recover, escalate).
- Produce a single technical-handoff doc pointing to the relevant
  scripts and parking tags by short SHA.
- All work in `reports/` and existing doc trees; no runtime code, no
  launchers, no DB inspection, no terminology data, no source docs.
- Carry the same default-off, review-bound, aggregate-only,
  privacy-clean invariants as PARK-26.

## 6. Recommended next 3-block sequence

1. **OPERATOR-MANUAL-CONSOLIDATION-01** — consolidate operator manual
   and technical handoff (G).
2. **PACKAGING-DEPLOYMENT-POLISH-03** — launcher/install/UX polish that
   references the consolidated manual (E).
3. **FREEZE-LOCAL-OPERATOR-RELEASE** — final release snapshot of the
   local operator artifact (H).

After step 3 lands, re-run a ROADMAP-04 audit before opening any
candidate from {A, B, C, D, F}. Cue expansion and additional
Unknown-track diagnostics remain off the menu.

## 7. Explicitly deferred work

- **MORE-UNKNOWN-DIAGNOSTICS (I)** — deferred. The residual-Unknown
  reduction track is parked at ~99.99% closed across DIAG-13A..21 with
  parking tags PARK-20..25. Reopen only if a future operator UAT block
  produces a new Unknown-related signal.
- **CUE-EXPANSION (J)** — explicitly NOT recommended. The standing
  project posture is unchanged: cue expansion adds risk to the
  classifier surface and licensing/privacy surface without improving any
  failing operator outcome.

## 8. Safety / privacy constraints

- No source documents opened.
- No raw OCR text, raw document text, raw filenames, private paths,
  PHI, secrets, DBs, backups, bundles, keys or terminology data read.
- No runtime DB contents inspected (only public-safe report metadata
  was read).
- No `app/main.py` / launcher / preflight / config modification.
- No tags created, moved, or deleted.
- No external APIs called or enabled.
- PARK-20..26 tags untouched.

## 9. Why cue expansion remains NOT recommended

Across DIAG-13..21 and ROADMAP-01..02, cue expansion has been ruled out
as a primary lever. It would re-open the classifier surface that the
residual-Unknown track explicitly stopped touching, expand the
regression surface, create new privacy/data-handling responsibilities,
and not resolve any currently failing operator outcome. ROADMAP-03
reaffirms: no cue expansion.

## 10. Why residual Unknown diagnostics remain parked

The 21-record residual Unknown surface (11 Sub-track A + 10 Sub-track B)
has been driven through DIAG-13A..21 with default-off metadata,
default-off render plan, env-gated Streamlit wiring, env-on aggregate
UAT and a Streamlit fixture audit. PARK-26 confirms no blocking bugs
and `hardening_change_needed=false`. Re-opening DIAG-22+ now would
re-touch a parked surface without changing any observable operator
outcome. Reopen only if a future operator UAT block records a fresh
Unknown-related failure.

## 11. Suggested Claude Code prompt title for the next block

`MEDAI-OPERATOR-MANUAL-CONSOLIDATION-01 — Consolidate Operator Manual and Technical Handoff`

## 12. Progress estimate

| Track | Before ROADMAP-03 | After ROADMAP-03 |
| --- | --- | --- |
| Whole MedAI project | ~93.0% done / ~7.0% remaining | ~93.0% done / ~7.0% remaining |

ROADMAP-03 is a planning audit and produces no runtime change, so the
project percentage does not advance. Forward motion resumes when the
recommended OPERATOR-MANUAL-CONSOLIDATION-01 block lands.
