# MEDAI-V2-FOUNDATION-DEFAULT-OFF-IMPLEMENTATION-PLAN-01

## Scope And Non-Scope

This is a reports-only implementation plan for a future V2 foundation default-off status registry. It defines the implementation target, file boundaries, public-safe contract, test expectations, rollback rules, and stop-on-failure gates before any code implementation begins.

No implementation begins in this block. The default-off status registry is not created. No runtime helper code, runtime wiring, app/main.py changes, Streamlit UI changes, launcher changes, startup/config changes, OCR routing, extraction logic, classifier logic, thresholds, parser behavior, fallback behavior, cue packs, DB/schema/migrations, persistence code, terminology/private adapter work, private data access, external APIs, or tags are changed.

## Prior V2 Planning And Parking Chain Summary

The plan follows the parked V2 sequence: architecture spec, foundation spec, runtime contracts, validation harness, UI shell spec, data-infra spec, extraction/OCR spec, ROADMAP-02, foundation implementation readiness, packaging spec, and ROADMAP-PARK-01. The carried-forward readiness outcome is `conditionally_ready_after_packaging_spec`. The carried-forward safest future implementation candidate is `V2 foundation default-off status registry`.

## Future Implementation Target

Target: `V2 foundation default-off status registry`.

Purpose:

- Provide a pure, import-safe, read-only registry of V2 feature/status flags.
- Record whether each V2 capability is `not_started`, `spec_only`, `typing_only`, `validation_only`, `planned_default_off`, `implemented_default_off`, `blocked`, `parked`, or `frozen`.
- Expose no runtime behavior.
- Control no runtime routing.
- Trigger no UI behavior.
- Touch no OCR, extraction, classifier, threshold, DB, terminology, clinical, or DDI logic.
- Serve only as a public-safe metadata surface for later reports and tests.

The future target must be default-off and non-runtime: no app/main.py import, no Streamlit import, no launcher import, no extraction/OCR import, no DB import, no private terminology import, no external package import, no environment variable read, no filesystem read, no network call, and no import side effects.

## Future File Boundaries

Allowed future implementation files:

- `clinical_knowledge/v2_foundation/status_registry.py`
- `clinical_knowledge/v2_foundation/__init__.py`
- `scripts/run_medai_v2_foundation_default_off_status_registry_01.py`
- `tests/test_medai_v2_foundation_default_off_status_registry_01.py`
- `reports/medai_v2_foundation_default_off_status_registry_01/*`

Disallowed future implementation files unless separately approved:

- `app/main.py`
- Streamlit UI files
- launchers
- startup/preflight/config files
- extraction/OCR/classifier/parser files
- DB/schema/migration/persistence files
- terminology/private adapter files
- clinical/DDI/cue-pack files
- runtime routing files

## Future Status Registry Contract

The future registry should contain only standard-library imports, frozen dataclasses, Enum values, tuples or dicts of controlled-vocabulary metadata, and pure functions returning copies or tuples of public-safe registry entries.

Future registry entry fields:

- `capability_id`
- `capability_name`
- `category`
- `status`
- `default_enabled`
- `runtime_wired`
- `ui_wired`
- `requires_operator_approval`
- `requires_privacy_review`
- `requires_safety_review`
- `blocked_reason`
- `source_spec_block`
- `public_report_safe`

Required defaults for implementation-sensitive entries:

- `default_enabled=false`
- `runtime_wired=false`
- `ui_wired=false`
- `requires_operator_approval=true`
- `public_report_safe=true`

Required categories: foundation, runtime_contracts, validation_harness, ui_shell, data_infra, extraction_ocr, packaging, terminology_private_adapter, cue_expansion, clinical_decision_logic.

Required statuses: frozen, parked, blocked, spec_only, typing_only, validation_only, planned_default_off, implemented_default_off, not_started.

## Future Registry Inventory

| Capability | Category | Status | Source anchor | Required posture |
| --- | --- | --- | --- | --- |
| v1_local_operator_release | foundation | frozen | 7ef8ffd | default off, not runtime wired |
| v2_architecture_spec | foundation | spec_only | 551af98 | public-safe metadata only |
| v2_foundation_spec | foundation | spec_only | 8b53d82 | public-safe metadata only |
| v2_runtime_contracts | runtime_contracts | typing_only | e6e33dd | public-safe metadata only |
| v2_validation_harness | validation_harness | validation_only | 73af6f6 | public-safe metadata only |
| v2_ui_shell | ui_shell | spec_only | 745a980 | public-safe metadata only |
| v2_data_infra | data_infra | spec_only | c4df477 | public-safe metadata only |
| v2_extraction_ocr | extraction_ocr | spec_only | d9ac47e | public-safe metadata only |
| v2_packaging | packaging | spec_only | 37d056a | public-safe metadata only |
| v2_foundation_status_registry | foundation | planned_default_off | this plan | default off, not runtime wired, not UI wired |
| terminology_private_adapter | terminology_private_adapter | blocked | operator gate | operator license verification incomplete |
| cue_expansion | cue_expansion | blocked | safety gate | explicitly not recommended |
| clinical_decision_logic_expansion | clinical_decision_logic | blocked | safety gate | requires separate clinical safety spec |

## Future Implementation Gates

A. Must remain standard-library-only.  
B. Must be import-safe and side-effect-free.  
C. Must not be imported by runtime/UI paths.  
D. Must not modify app/main.py.  
E. Must not modify Streamlit.  
F. Must not modify launchers/startup/config.  
G. Must not modify OCR/extraction/classifier/threshold/parser/fallback/cue logic.  
H. Must not modify DB/schema/migrations/persistence.  
I. Must not touch terminology/private adapter logic.  
J. Must not read private data or runtime DB contents.  
K. Must not use external APIs.  
L. Must include focused tests for registry defaults and blocked statuses.  
M. Must pass public-report privacy checks.  
N. Must pass prior V2 regression pack.  
O. Must pass V1 health validations.

## Future Rollback / Stop Rules

Stop if future registry import prints stdout/stderr, imports non-stdlib modules, is imported by app/main.py or runtime files, defaults any registry entry to enabled without approval, unblocks terminology/private adapter or cue expansion, leaks private content in public reports, changes runtime behavior, stages files outside the allowed scope, fails V1 validations, or fails prior V2 tests.

## Blocked And Deferred Track Status

Private adapter implementation, real private-store access, licensed row reads, private license acknowledgement contents, runtime DB migration, extraction/OCR behavior changes, UI implementation, packaging implementation, launcher changes, direct V2 implementation, terminology-driven clinical inference, and cue expansion remain blocked or deferred. Cue expansion remains explicitly not recommended.

## Safety And Privacy Invariant Summary

This block changes no runtime behavior and accesses no private data. Source documents, raw OCR text, extracted text, filenames, private paths, PHI, secrets, runtime DB rows, licensed terminology rows, private config contents, and license acknowledgement contents are not read. External APIs are not used. No tags are created, moved, deleted, or repointed.

## Validation Matrix

Validation includes focused default-off implementation plan tests, prior V2 roadmap parking/packaging/readiness/roadmap/extraction/data-infra/UI-shell/validation-harness/runtime-contract tests, implementation-plan audit script, public report privacy checks, final CKA MVP validation, B07 validation, ROUTE-FIX validation, UI ops validation, UI boot validation, and staged safety check.

## Recommended Next 3-Block Sequence

1. `V2-FOUNDATION-DEFAULT-OFF-STATUS-REGISTRY-01`
2. `V2-ROADMAP-03`
3. `V2-ROADMAP-PARK-02_OR_RELEASE-FREEZE-SNAPSHOT`

## Final Recommendation

Recommended next block: `V2-FOUNDATION-DEFAULT-OFF-STATUS-REGISTRY-01`.

This is a planning-only block. This block creates only public planning artifacts. No registry code, runtime wiring, application entrypoint edits, launch changes, startup changes, config changes, OCR changes, extraction changes, DB changes, schema changes, migration changes, tags, terminology work, private adapter work, or cue expansion were created. The frozen V1 release remains preserved.
