# MEDAI-V2-ROADMAP-02

## Scope And Non-Scope

This is a reports-only post-V2 planning re-rank audit. It reviews the completed V2 planning sequence and selects the safest next strategic posture. It does not start V2 implementation.

No runtime code, UI code, extraction logic, OCR routing, classifier logic, thresholds, parser behavior, fallback behavior, cue packs, DB schema, migrations, persistence code, launchers, startup/config, terminology/private adapter code, or external APIs are changed.

## Prior V2 Planning Sequence Summary

| Block | Summary | Runtime posture |
| --- | --- | --- |
| MEDAI-V2-ARCHITECTURE-SPEC-01 | Architecture direction established. | Reports-only; no runtime behavior changed. |
| MEDAI-V2-FOUNDATION-SPEC-02 | Foundation doctrine, invariants, and stop-on-failure rules established. | Reports-only. |
| MEDAI-V2-RUNTIME-CONTRACTS-01 | Typing-only runtime contracts created. | No concrete adapters; no runtime wiring. |
| MEDAI-V2-VALIDATION-HARNESS-01 | V1 health-check catalog preserved and V2 validation matrix created. | Contract-conformance harness only. |
| MEDAI-V2-UI-SHELL-SPEC-01 | Operator UI shell planned. | No Streamlit implementation, action widgets, callbacks, or session-state changes. |
| MEDAI-V2-DATA-INFRA-SPEC-01 | Data and persistence architecture planned. | Runtime-DB-row-blind doctrine; no schema, migration, or DB access. |
| MEDAI-V2-EXTRACTION-SPEC-01 | Extraction and OCR architecture planned. | No OCR routing, extraction, classifier, threshold, parser, fallback, or cue changes. |

## Validation Health Summary

The preceding V2 extraction block completed focused V2 extraction tests, prior V2 regression tests, report privacy checks, final CKA MVP validation, B07 term01 validation, ROUTE-FIX validation, UI ops validation, UI boot validation, and staged safety checks. This roadmap block relies only on public-safe reports and typing-only modules.

## Remaining Workstream Ranking

| Rank | Workstream | Readiness | Risk | Recommended posture |
| --- | --- | --- | --- | --- |
| 1 | V2-FOUNDATION-IMPLEMENTATION-READINESS-01 | High | Low to moderate | Run a reports-only readiness audit before any default-off implementation. |
| 2 | V2-PACKAGING-SPEC-01 | High | Low to moderate | Define operator packaging and deployment expectations. |
| 3 | V2-ROADMAP-PARK-01 | High | Low | Park the completed planning sequence if release hygiene is preferred. |
| 4 | V2-CONTRACT-STUB-EXPANSION-01 | Medium | Low | Use only if readiness finds missing typing seams. |
| 5 | V2-UI-IMPLEMENTATION-READINESS-01 | Medium | Moderate | Audit UI implementation readiness without UI code changes. |
| 6 | V2-DATA-MIGRATION-READINESS-01 | Medium | Moderate | Audit migration readiness without DB access or schema changes. |
| 7 | V2-EXTRACTION-IMPLEMENTATION-READINESS-01 | Medium | High | Defer until foundation readiness and stricter extraction safety gates pass. |
| 8 | V2-PRIVATE-TERMINOLOGY-WAIT-GATE | Blocked | High | Keep blocked pending operator license confirmation. |
| 9 | V2-ROADMAP-03 | Later | Low | Use as checkpoint after readiness or packaging work. |
| 10 | V2-CUE-EXPANSION | Not recommended | High | Keep closed. |

## Risk Matrix

| Workstream | Implementation risk | Safety/privacy risk | Validation cost | Dependency risk | Operator value |
| --- | --- | --- | --- | --- | --- |
| V2-FOUNDATION-IMPLEMENTATION-READINESS-01 | Low to moderate | Low | Moderate | Low | High |
| V2-PACKAGING-SPEC-01 | Low | Low | Low | Low | Medium-high |
| V2-ROADMAP-PARK-01 | Low | Low | Low | Low | Medium |
| V2-CONTRACT-STUB-EXPANSION-01 | Low | Low | Low to moderate | Medium | Medium |
| V2-UI-IMPLEMENTATION-READINESS-01 | Moderate | Low to moderate | Moderate | Medium | High |
| V2-DATA-MIGRATION-READINESS-01 | Moderate | Moderate | Moderate-high | Medium | Medium |
| V2-EXTRACTION-IMPLEMENTATION-READINESS-01 | High | Moderate | High | High | High |
| V2-PRIVATE-TERMINOLOGY-WAIT-GATE | High | High | High | High | Deferred |
| V2-CUE-EXPANSION | High | High | High | High | Not recommended |

## Blocked And Deferred Work Status

Blocked until a separate operator-approved spec changes the status:

- private adapter implementation
- real private-store access
- licensed terminology row reads
- private license-acknowledgement contents access
- external terminology runtime APIs
- MeSH integration until license confirmation and completed operator-side download
- terminology-driven DDI, diagnosis, treatment, and medication inference
- runtime private terminology output beyond parked helper/wiring
- runtime DB migration
- extraction/OCR behavior changes
- UI implementation
- clinical decision logic expansion

Explicitly deferred or not recommended:

- MORE-UNKNOWN-DIAGNOSTICS unless a fresh failure signal exists
- CUE-EXPANSION
- PRIVATE-ADAPTER-IMPLEMENTATION
- direct V2 implementation

Cue expansion remains explicitly not recommended.

## Recommended Next Posture

Top recommendation: `V2-FOUNDATION-IMPLEMENTATION-READINESS-01`.

Reason: the completed V2 planning sequence is broad enough to justify a reports-only readiness audit before any default-off implementation planning. This is not a recommendation to implement V2 directly. It is a checkpoint to decide whether implementation prerequisites, rollback plans, validation gates, and safety boundaries are complete.

## Recommended Next 3-Block Sequence

1. `V2-FOUNDATION-IMPLEMENTATION-READINESS-01`
2. `V2-PACKAGING-SPEC-01`
3. `V2-ROADMAP-PARK-01`

## Safety And Privacy Invariant Summary

This block changed no runtime behavior and accessed no private data. Source documents, raw OCR text, extracted text, filenames, private paths, PHI, secrets, runtime DB rows, licensed terminology rows, private config contents, and license acknowledgement contents were not read. External APIs were not used.

## Freeze And Parking Strategy

If readiness finds any safety, privacy, validation, rollback, dependency, or operator uncertainty, the next posture should shift to `V2-ROADMAP-PARK-01` or freeze-maintenance-only. V1 frozen release remains preserved.

## Validation Matrix

This roadmap audit must be validated by focused roadmap tests, prior V2 extraction/data-infra/UI-shell/validation-harness/runtime-contract tests, the roadmap audit script, public report privacy checks, final CKA MVP validation, B07 term01 validation, ROUTE-FIX validation, UI ops validation, UI boot validation, and staged safety checks.

## Final Recommendation

Direct V2 implementation remains out of scope. Terminology/private adapter implementation and cue expansion remain closed. Extraction/OCR behavior, DB/schema/migrations, and UI changes require separate guarded blocks with rollback plans. The frozen V1 release baseline remains preserved.
