# MEDAI-ROADMAP-02

## Executive recommendation

The highest-value next implementation block is `REAL-WORLD-OPERATOR-UAT-02`.

Reason: launch readiness, local operator handoff, UI boot, UI ops, final CKA MVP, B07, and ROUTE-FIX validations are already complete. The most useful next step is to prove the handoff in a controlled operator workflow using safe synthetic or approved local test inputs, while keeping runtime behavior unchanged unless a blocking UI/reporting defect is found.

## Current readiness baseline

- RELEASE-HANDOFF-01 is complete and ready.
- PACKAGING-LAUNCHER-HARDEN-01 is complete and ready.
- OPERATOR-UAT-01 is complete and ready.
- PARK-25 parked the DIAG-21 Streamlit fixture audit.
- PDF text/layout quality track is parked.
- Residual Unknown micro-diagnostics are parked.
- Local-only and external-API-disabled posture is preserved.

## Candidate workstream ranking

| Workstream | Operator value | Implementation risk | Safety/privacy risk | Validation cost | Practical usefulness | Dependency risk | Recommended order |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| REAL-WORLD-OPERATOR-UAT-02 | 5 | 2 | 2 | 3 | 5 | 2 | 1 |
| UI-USABILITY-POLISH-01 | 5 | 3 | 1 | 3 | 5 | 3 | 2 |
| DATA-RUNTIME-HARDEN-01 | 4 | 3 | 3 | 3 | 4 | 3 | 3 |
| PACKAGING-POLISH-02 | 3 | 1 | 1 | 2 | 4 | 1 | 4 |
| BROADER-V2-ARCHITECTURE-01 | 5 | 5 | 3 | 5 | 4 | 5 | 5 |
| REAL-CORPUS-VALIDATION-02 | 3 | 2 | 4 | 4 | 3 | 3 | 6 |
| MORE-UNKNOWN-DIAGNOSTICS | 1 | 2 | 2 | 3 | 1 | 2 | defer |
| CUE-EXPANSION | 1 | 4 | 3 | 4 | 1 | 4 | not recommended |

## Top recommended next block

`REAL-WORLD-OPERATOR-UAT-02 — Controlled Local Operator Workflow UAT`

Suggested scope:

- Use only safe test folders and synthetic inputs, or operator-approved local test inputs.
- Exercise the documented handoff path end to end.
- Confirm startup, Run & Review availability, validation commands, review-bound status, local-only posture, and recovery guidance.
- Produce a receipt-only report with no raw text, filenames, private paths, PHI, secrets, source documents, DBs, or runtime artifacts.
- Keep OCR, extraction, classifier, thresholds, cue packs, clinical logic, DDI, launchers, and external API posture unchanged.

## Recommended next 3-block sequence

1. `REAL-WORLD-OPERATOR-UAT-02 — Controlled Local Operator Workflow UAT`
2. `UI-USABILITY-POLISH-01 — Operator friction fixes from UAT only`
3. `DATA-RUNTIME-HARDEN-01 — Startup, config, DB, signing, and credential hygiene audit`

## Workstreams to defer

- `MORE-UNKNOWN-DIAGNOSTICS`
- `CUE-EXPANSION`
- `BROADER-V2-ARCHITECTURE-01` until UAT identifies concrete product constraints.
- `REAL-CORPUS-VALIDATION-02` until privacy handling and operator need justify the validation cost.

## Safety and privacy constraints

- Keep all work local-only.
- Keep external APIs disabled.
- Keep private/source documents closed unless a later block explicitly authorizes safe handling.
- Keep raw OCR text, raw document text, filenames, private paths, PHI, and secrets out of reports.
- Keep lab value and medication parsing out of scope.
- Keep DDI and clinical interpretation out of scope.
- Keep affected records review-bound.

## Why cue expansion remains not recommended

Cue expansion has low marginal value after the parked Unknown-reduction track and carries false-positive risk. It should remain blocked unless a future operator UAT produces a specific, privacy-safe, repeatable gap that cannot be solved by UI, handoff, or workflow changes.

## Why residual Unknown diagnostics should remain parked

The residual Unknown track is already estimated at approximately 99.995% complete. Continuing it now would likely produce diminishing returns compared with validating the actual operator handoff and identifying practical workflow friction.

## Suggested Codex prompt title for next block

`MEDAI-REAL-WORLD-OPERATOR-UAT-02 — Controlled Local Operator Workflow UAT`

## Progress estimate

Whole MedAI project estimate remains approximately 92.3%.
