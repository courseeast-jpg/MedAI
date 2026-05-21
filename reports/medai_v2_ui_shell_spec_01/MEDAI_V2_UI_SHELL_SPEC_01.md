# MEDAI-V2-UI-SHELL-SPEC-01 — Short Summary

Reports-only V2 operator UI shell planning SPEC. 8 shells, 10 future
implementation gates, read-only doctrine carried forward from
DIAG-19. No Streamlit code change. No `app/main.py` change. No UI
buttons / callbacks / session-state. No runtime behavior change.

## Deliverables

- `scripts/run_medai_v2_ui_shell_spec_01.py` — reports-only audit.
- `tests/test_medai_v2_ui_shell_spec_01.py` — focused tests.
- 3 public-safe reports.

## Shells (8)

| Shell | Purpose |
| --- | --- |
| S1 Operator Home / Current Run | Show current local session state |
| S2 Run & Review | Operator review of processed documents |
| S3 Advanced Technical Details | Read-only diagnostic visibility |
| S4 Safety / Privacy Status | Visible local-only / external-API-blocked / privacy-audit / review-bound status |
| S5 Validation / Health | Show V1 five-validation + future V2 matrix status (read-only) |
| S6 Terminology / Clinical Knowledge | Show terminology / private-adapter wait gate status only |
| S7 Parking / Release | Show frozen release + parked track anchors |
| S8 Operator Guidance | Plain-language operator guidance |

No shell adds action buttons, callbacks, or session-state in this
SPEC. 0 / 0 / 0 in the inventory.

## Verdict

| Field | Value |
| --- | --- |
| `block_mode` | `reports_only_ui_shell_spec` |
| `requires_preceding_spec` | true |
| `preceding_architecture_spec_present` | true (`551af98`) |
| `preceding_foundation_spec_present` | true (`8b53d82`) |
| `preceding_runtime_contracts_present` | true (`e6e33dd`) |
| `preceding_validation_harness_present` | true (`73af6f6`) |
| `ui_shell_spec_created` | true |
| `screen_inventory_created` | true |
| `future_ui_implementation_gates_defined` | true |
| `streamlit_code_changed` | false |
| `app_main_changed` | false |
| `no_ui_actions_added` | true |
| `no_callbacks_added` | true |
| `no_session_state_logic_added` | true |
| `v1_release_preserved` | true |
| `cue_expansion_recommended` | false |
| Next recommended block | **`V2-DATA-INFRA-SPEC-01`** |

## Doctrine

Cue expansion remains explicitly **NOT** recommended. Private adapter
implementation remains blocked. V1 frozen release at `7ef8ffd`
continues to be the durable shipped artifact.

## Progress

- Whole MedAI project: **~96.5%** done / ~3.5% remaining.
