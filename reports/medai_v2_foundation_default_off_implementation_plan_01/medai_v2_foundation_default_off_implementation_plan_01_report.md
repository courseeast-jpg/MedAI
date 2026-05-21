# MEDAI-V2-FOUNDATION-DEFAULT-OFF-IMPLEMENTATION-PLAN-01 Report

## Scope And Non-Scope

Reports-only implementation plan. No status registry, default-off helper, runtime helper, runtime wiring, UI code, launcher, startup/config, extraction/OCR, DB, terminology/private adapter, external API, or tag change is created.

## Prior V2 Planning And Parking Chain Summary

The plan follows the parked V2 chain through ROADMAP-PARK-01 and carries forward `conditionally_ready_after_packaging_spec` plus the safest future candidate `V2 foundation default-off status registry`.

## Future Implementation Target

`V2 foundation default-off status registry`

The target is a pure, import-safe, read-only metadata registry for V2 capability statuses. It must expose no runtime behavior, control no routing, trigger no UI behavior, read no environment variables, touch no filesystem, call no network, and import no non-standard-library packages.

## Future File Boundaries

Allowed future files are limited to `clinical_knowledge/v2_foundation/status_registry.py`, `clinical_knowledge/v2_foundation/__init__.py`, the matching report script, focused tests, and public-safe reports. app/main.py, Streamlit files, launchers, startup/config, extraction/OCR/classifier/parser, DB/persistence, terminology/private adapter, clinical/DDI/cue, and runtime routing files remain disallowed unless separately approved.

## Future Status Registry Contract

The future contract uses standard-library imports, frozen dataclasses, Enums, controlled-vocabulary metadata, and pure functions returning public-safe copies or tuples. Required entry fields include capability id/name, category, status, default-enabled flag, runtime/UI wiring flags, approval and review flags, blocked reason, source spec block, and public-report-safe flag.

## Future Registry Inventory

Initial entries cover the frozen V1 local operator release, V2 architecture/foundation/runtime-contracts/validation/UI/data/extraction/packaging planning chain, the planned default-off status registry, blocked terminology private adapter, blocked cue expansion, and blocked clinical decision logic expansion.

## Future Implementation Gates

The future implementation must remain standard-library-only, import-safe, side-effect-free, not imported by runtime/UI paths, and must not modify app/main.py, Streamlit, launchers, startup/config, extraction/OCR/classifier/threshold/parser/fallback/cue logic, DB/schema/migrations/persistence, terminology/private adapter logic, private data, runtime DB contents, or external APIs. It must pass focused tests, privacy checks, prior V2 regression tests, and V1 health validations.

## Future Rollback / Stop Rules

Stop on import output, non-stdlib imports, runtime/UI imports, default-enabled entries without approval, unblocked terminology/private adapter, unblocked cue expansion, private report content, runtime behavior change, unexpected staged files, V1 validation failure, or prior V2 test failure.

## Blocked And Deferred Track Status

Private adapter implementation, real private-store access, licensed row reads, private license acknowledgement contents, runtime DB migration, extraction/OCR behavior changes, UI implementation, packaging implementation, launcher changes, direct V2 implementation, terminology-driven clinical inference, and cue expansion remain blocked or deferred. Cue expansion remains explicitly not recommended.

## Safety And Privacy Invariant Summary

No runtime behavior changed. No private data, source documents, raw OCR text, extracted text, filenames, private paths, runtime DB rows, licensed terminology rows, private configs, license acknowledgement contents, PHI, or secrets were accessed. External APIs were not used. No tags were created, moved, deleted, or repointed.

## Validation Matrix

Validation includes focused implementation-plan tests, prior V2 roadmap parking/packaging/readiness/roadmap/extraction/data-infra/UI-shell/validation-harness/runtime-contract tests, implementation-plan audit script, public report privacy checks, final CKA MVP validation, B07 validation, ROUTE-FIX validation, UI ops validation, UI boot validation, and staged safety check.

## Recommended Next 3-Block Sequence

1. `V2-FOUNDATION-DEFAULT-OFF-STATUS-REGISTRY-01`
2. `V2-ROADMAP-03`
3. `V2-ROADMAP-PARK-02_OR_RELEASE-FREEZE-SNAPSHOT`

## Final Recommendation

Create the actual default-off status registry only in the next separately approved implementation block. The frozen V1 release remains the preserved shipped baseline.

## Validation Results

Focused V2 default-off implementation-plan tests passed: 9 tests. The implementation-plan audit script passed. The prior V2 regression pack passed: 133 tests. Public report privacy checks passed for all three new reports. Final CKA MVP validation passed 12 of 12 cases with 693 tests and external API use false. B07 term01 validation passed 6 of 6 cases with external API use false. ROUTE-FIX, UI ops, and UI boot validations passed. Staged safety passed with only V2 default-off implementation-plan report, script, and test files staged.
