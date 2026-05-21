# MEDAI-V2-PACKAGING-SPEC-01

## Scope And Non-Scope

This is a reports-only V2 packaging and deployment planning spec. It defines future packaging, launch, operator handoff, validation receipt, and release hygiene boundaries before any packaging implementation occurs.

No packaging implementation begins in this block. No launchers, installer scripts, deployment scripts, runtime code, app/main.py, Streamlit UI code, startup/config, OCR routing, extraction logic, classifier logic, thresholds, parser behavior, fallback behavior, cue packs, DB/schema/migrations, persistence code, terminology/private adapter work, private data handling, external APIs, or tags are changed.

## Prior V2 Dependency Chain

| Dependency | Packaging relevance |
| --- | --- |
| MEDAI-V2-ARCHITECTURE-SPEC-01 | Establishes the V2 planning direction. |
| MEDAI-V2-FOUNDATION-SPEC-02 | Defines invariants, stop-on-failure rules, and block taxonomy. |
| MEDAI-V2-RUNTIME-CONTRACTS-01 | Provides typing-only runtime contracts without concrete adapters. |
| MEDAI-V2-VALIDATION-HARNESS-01 | Preserves V1 health checks and V2 validation matrix expectations. |
| MEDAI-V2-UI-SHELL-SPEC-01 | Defines operator UI shell boundaries without Streamlit implementation. |
| MEDAI-V2-DATA-INFRA-SPEC-01 | Defines data/persistence boundaries and runtime-DB-row-blind doctrine. |
| MEDAI-V2-EXTRACTION-SPEC-01 | Defines extraction/OCR boundaries without behavior changes. |
| MEDAI-V2-ROADMAP-02 | Selects readiness-before-implementation posture. |
| MEDAI-V2-FOUNDATION-IMPLEMENTATION-READINESS-01 | Carries forward `conditionally_ready_after_packaging_spec`. |

## Local Operator Packaging Boundary

The frozen V1 local operator release remains the preserved shipped baseline. V2 packaging is planning-only here. Future V2 packages must preserve local-only defaults, review-bound posture, external-API-blocked defaults, no auto-accept, and public-safe validation evidence. This block changes no launchers, installer files, or release artifacts.

## Launcher Boundary

Existing launchers remain unchanged. Future launcher changes require a separate implementation block with UI boot validation, UI ops validation, staged safety checks, public-report privacy checks, and a rollback path. Any launcher behavior change must include public-safe operator recovery steps and must not print private paths, raw filenames, secrets, PHI, or runtime DB rows.

## Startup / Preflight Boundary

No startup or preflight logic changes occur in this block. Future startup/preflight changes must be default-safe, must not mask failures, and must preserve operator-visible diagnostics without leaking private paths, raw filenames, secrets, PHI, source content, licensed rows, private config contents, or runtime DB rows.

## Validation Receipt Boundary

Packaging must carry the V1 five-validation health-check set:

- Final CKA MVP validation
- B07 term01 opt-in integration
- ROUTE-FIX 01
- UI ops panel
- UI boot fix

Packaging must also carry focused V2 validation packs for each V2 planning or implementation block. Receipts must remain public-report-safe, aggregate-only, and explicit about whether external APIs, private data access, runtime DB access, tags, and behavior changes occurred.

## Release Artifact Boundary

Future allowed release artifacts:

- public-safe markdown reports
- public-safe JSON receipts
- operator manuals
- technical handoff docs
- launcher documentation
- validation summary receipts
- parking/freeze snapshots
- optional source bundle only after privacy and staged-safety checks pass

Prohibited release artifacts:

- source/private documents
- runtime DB files
- raw OCR/text
- raw filenames
- private paths
- PHI
- secrets, API keys, or credentials
- private config contents
- LICENSE_ACK_PRIVATE.json contents
- licensed terminology rows
- terminology data folders
- backups containing private data

## Environment Boundary

V2 packaging must preserve local-only defaults. External API use remains blocked by default. Any external dependency must be explicitly optional, disabled by default, and gated. No cloud deployment is approved by this spec. Any future cloud packaging requires a separate privacy/safety spec.

## Operator Handoff Boundary

Future operator handoff materials must include:

- plain-language startup steps
- health-check steps
- safe shutdown steps
- recovery steps
- prohibited-actions section
- privacy limits
- local-only meaning
- review-bound meaning
- blocked tracks list
- known sandbox limitations
- current safe shipped baseline pointer

Handoff materials must be public-safe and must not include private paths, raw filenames, source content, PHI, secrets, runtime DB rows, private config contents, or licensed row content.

## Parking / Freeze Boundary

The frozen V1 release remains preserved. Existing park/freeze tags must not move. V2 planning can be parked after this block if desired. Any future V2 implementation plan must reference this packaging spec and the foundation implementation readiness audit.

## Future Packaging Implementation Gates

A. Launcher changes require a separate implementation block.  
B. Installer or package artifact changes require staged safety and public-report privacy checks.  
C. Startup/preflight changes require focused tests and rollback.  
D. Cloud deployment work requires a separate privacy/safety spec.  
E. External API configuration must remain optional, default-off, and gated.  
F. Source bundles must exclude private docs, runtime DBs, terminology data, private configs, secrets, backups, and licensed rows.  
G. Operator manuals must be public-safe and omit private paths.  
H. Release/freeze tag creation must occur only in a separate parking/freeze block.  
I. Packaging changes must preserve the frozen V1 release baseline.  
J. Cue expansion remains explicitly not recommended and out of scope.

## Safety And Privacy Invariants

This block changes no runtime behavior and accesses no private data. Source documents, raw OCR text, extracted text, filenames, private paths, PHI, secrets, runtime DB rows, licensed terminology rows, private config contents, and license acknowledgement contents are not read. External APIs are not used. No tags are created or touched.

## Validation Matrix

Validation includes focused V2 packaging spec tests, prior V2 foundation implementation readiness tests, roadmap-02 tests, extraction/data-infra/UI-shell/validation-harness/runtime-contract tests, packaging audit script, public report privacy checks, final CKA MVP validation, B07 validation, ROUTE-FIX validation, UI ops validation, UI boot validation, and staged safety check.

## Recommended Next Block

`V2-ROADMAP-PARK-01`

Then either `V2-FOUNDATION-DEFAULT-OFF-IMPLEMENTATION-PLAN-01_OR_FREEZE-MAINTENANCE-ONLY`, followed by `V2-ROADMAP-03`.

V2 packaging implementation must not start directly. Launchers, startup/preflight/config, deployment automation, release tags, terminology/private adapter implementation, and cue expansion remain unchanged. The frozen V1 release remains the preserved shipped baseline.
