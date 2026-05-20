# MEDAI-ROADMAP-05

## Executive Recommendation

Recommended next phase: `CKA-TERM-INTEGRATION-PARK-01`.

The terminology helper and UI wiring mini-track is complete and validated. The safest immediate next step is to park that chain with a reports/tags snapshot before opening any private terminology adapter, broader product work, or v2 architecture work.

## Current Terminology Mini-Track Status

- `CKA-TERM-INTEGRATION-PLAN-01`: complete.
- `CKA-TERM-INTEGRATION-NEXT-01`: default-off helper complete.
- `CKA-TERM-INTEGRATION-UAT-01`: synthetic helper UAT complete.
- `CKA-TERM-INTEGRATION-WIRING-NEXT-01`: default-off read-only Advanced technical details wiring complete.
- `CKA-TERM-INTEGRATION-WIRING-UAT-01`: synthetic UI wiring UAT complete.

## Current Frozen Release Baseline

The local operator release remains frozen at `7ef8ffd`. Freeze and PARK tags remain untouched. The new terminology helper chain did not reopen the frozen release by default because all helper and UI surfaces remain default-off and review-bound.

## Candidate Next-Phase Ranking

| Order | Candidate | Medical value | Operator value | Compliance value | Implementation risk | Safety/privacy risk | Validation cost | Dependency risk | Reversibility |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `CKA-TERM-INTEGRATION-PARK-01` | 3 | 3 | 5 | 1 | 1 | 1 | 1 | 5 |
| 2 | `CKA-TERM-LICENSE-GATE-SPEC-02` | 4 | 2 | 5 | 2 | 3 | 3 | 3 | 5 |
| 3 | `CKA-TERM-INTEGRATION-PRIVATE-ADAPTER-SPEC-01` | 5 | 3 | 5 | 3 | 4 | 4 | 4 | 4 |
| 4 | `FREEZE-MAINTENANCE-ONLY` | 2 | 4 | 5 | 1 | 1 | 1 | 1 | 5 |
| 5 | `V2-ARCHITECTURE-SPEC-01` | 4 | 3 | 4 | 2 | 2 | 4 | 4 | 5 |
| 6 | `PRODUCT-UX-NEXT-01` | 2 | 4 | 2 | 3 | 2 | 3 | 2 | 4 |
| 7 | `DATA-INFRA-NEXT-02` | 2 | 3 | 3 | 3 | 3 | 3 | 3 | 4 |
| 8 | `REAL-CORPUS-VALIDATION-03` | 4 | 3 | 3 | 3 | 5 | 5 | 4 | 4 |
| 9 | `CKA-TERM-INTEGRATION-PRIVATE-ADAPTER-NEXT-01` | 5 | 3 | 3 | 5 | 5 | 5 | 5 | 2 |
| 10 | `MORE-UNKNOWN-DIAGNOSTICS` | 1 | 1 | 2 | 2 | 2 | 3 | 2 | 4 |
| 11 | `CUE-EXPANSION` | 1 | 1 | 1 | 5 | 5 | 5 | 5 | 2 |

## Top Recommended Next Phase

`CKA-TERM-INTEGRATION-PARK-01`

Rationale: the terminology chain is now complete through SPEC, helper, helper UAT, UI wiring, and UI wiring UAT. Parking it gives the project a clean provenance checkpoint before any work touches licensing gates, private terminology adapters, or broader architecture.

## Recommended Next 3-Block Sequence

1. `CKA-TERM-INTEGRATION-PARK-01`
2. `CKA-TERM-LICENSE-GATE-SPEC-02`
3. `CKA-TERM-INTEGRATION-PRIVATE-ADAPTER-SPEC-01`

## Explicitly Deferred Work

- `MORE-UNKNOWN-DIAGNOSTICS`
- `CUE-EXPANSION`

## Safety And Privacy Constraints

ROADMAP-05 is reports-only. It did not modify runtime code, `app/main.py`, helper code, Streamlit wiring, launchers, startup preflight, config, extraction, OCR, classifier behavior, thresholds, cue packs, DDI, clinical logic, or external API behavior.

No source/private documents, runtime DB contents, licensed terminology rows, `LICENSE_ACK_PRIVATE.json` contents, terminology data folders, raw text, raw filenames, private paths, PHI, or secrets were opened or reported.

## Why Cue Expansion Remains Not Recommended

The terminology mini-track proves safe default-off metadata and read-only UI visibility without touching classifier behavior. Cue expansion would reopen the classifier surface, increase regression cost, and add safety/privacy risk without a current failure signal.

## Why Residual Unknown Diagnostics Remain Parked

The residual Unknown diagnostics track is already parked. The terminology chain did not create a new Unknown-related operational failure, so reopening Unknown diagnostics would be speculative.

## Suggested Prompt Title For Next Block

`MEDAI-CKA-TERM-INTEGRATION-PARK-01 — Park Default-Off Terminology Helper and UI Wiring Chain`

## Progress Estimate

Whole MedAI project estimate: approximately 94.7%.
