# MEDAI-V2-FOUNDATION-SPEC-02 — Short Summary

Reports-only v2 foundation control SPEC. Converts the v2 architecture
plan into durable foundation doctrine.

## Decision

| Field | Value |
| --- | --- |
| `foundation_doctrine_created` | **true** |
| `stop_on_failure_rules_defined` | **true** |
| `future_block_taxonomy_defined` | **true** |
| `v2_implementation_started` | **false** |
| `v1_release_preserved` | **true** |
| Next recommended block | **`MEDAI-V2-RUNTIME-CONTRACTS-01`** |

## Foundation principles

- v1 frozen release at `7ef8ffd` is the safe baseline.
- v2 is contract-first / SPEC-first; no implementation without a
  preceding SPEC.
- Local-only, review-bound, default-off posture preserved.
- Clinical behavior remains safety-gated.
- Cue expansion remains explicitly **NOT** recommended.

## Required invariants (per-block)

`local_only_default`, `external_api_default_blocked`,
`review_bound_default`, `no_auto_accept_without_spec`,
`no_private_data_access_without_spec`,
`no_licensed_row_access_without_operator_license_gate`,
`no_runtime_db_migration_without_rollback_spec`,
`no_clinical_inference_expansion_without_safety_spec`,
`cue_expansion_recommended: false`, `v1_release_preserved: true`,
`aggregate_only_public_reports_required`,
`public_report_privacy_check_required_before_commit`,
`staged_safety_check_required_before_commit`,
`focused_pytest_module_required_per_implementation_block`.

## Allowed block classes (9)

`reports_only_spec`, `contract_stub_only`, `validation_harness_only`,
`default_off_helper`, `default_off_ui_surface`,
`data_migration_spec_only`, `operator_uat_receipt`,
`parking_snapshot`, `release_freeze_snapshot`.

## Disallowed block classes (11, unless separately approved)

`direct_runtime_rewrite`, `direct_extraction_behavior_change`,
`direct_ocr_routing_change`, `direct_threshold_change`,
`direct_classifier_behavior_change`, `direct_cue_pack_expansion`,
`direct_private_adapter_implementation`,
`direct_licensed_terminology_row_access`,
`direct_external_api_runtime_integration`,
`direct_clinical_decision_logic_expansion`,
`direct_runtime_db_migration`.

## Frozen / parked track anchors

| Track | Commit |
| --- | --- |
| Local operator release | `7ef8ffd` |
| Residual Unknown reduction | `3e46461` |
| Text-layer eval spec | `9f9e22d` |
| PDF text/layout default-off | `f4d3cc6` |
| PDF text/layout Streamlit wiring | `748c32a` |
| DIAG-20 operator UAT | `1b14ffe` |
| DIAG-21 fixture audit | `6b31678` |
| Operator readiness + runtime hardening | `91b9eba` |
| Terminology helper/wiring mini-track | `e398a75` |
| License-gated private adapter track | `b9b19ad` |
| Private terminology config boundary | `60f1114` (complete & verified) |
| MeSH local helper / download | `cedbbd3` (blocked operator-side) |
| Manual license verification gate | `376ca4e` (blocked operator-side) |

## Recommended next 3-block sequence

1. `MEDAI-V2-RUNTIME-CONTRACTS-01`
2. `MEDAI-V2-VALIDATION-HARNESS-01`
3. `MEDAI-V2-UI-SHELL-SPEC-01` or `MEDAI-V2-DATA-INFRA-SPEC-01`

## Progress

- Whole MedAI project: **~96.2%** done / ~3.8% remaining.
