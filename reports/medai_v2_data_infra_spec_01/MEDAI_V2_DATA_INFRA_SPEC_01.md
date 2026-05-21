# MEDAI-V2-DATA-INFRA-SPEC-01 — Short Summary

Reports-only V2 data / persistence architecture SPEC. 8 conceptual
stores, ledger / audit separation, rollback doctrine, 9 migration
gates, public-report data doctrine, 10 future implementation gates.
No DB schema change. No migration. No runtime DB row read. No
persistence code change.

## Deliverables

- `scripts/run_medai_v2_data_infra_spec_01.py` — reports-only audit.
- `tests/test_medai_v2_data_infra_spec_01.py` — focused tests.
- 3 public-safe reports.

## Conceptual store map (8)

| ID | Store |
| :-: | --- |
| PS1 | `document_registry_metadata_store` |
| PS2 | `extraction_result_store` |
| PS3 | `review_queue_store` |
| PS4 | `operator_action_ledger` |
| PS5 | `validation_receipt_store` |
| PS6 | `audit_observability_event_store` |
| PS7 | `terminology_aggregate_cache_if_later_approved` (blocked) |
| PS8 | `quarantine_blocked_item_store_if_later_approved` |

All stores conceptual only. No schema / migration / persistence code.

## Verdict

| Field | Value |
| --- | --- |
| `block_mode` | `reports_only_data_infra_spec` |
| `requires_preceding_spec` | true |
| `preceding_architecture_spec_present` | true (`551af98`) |
| `preceding_foundation_spec_present` | true (`8b53d82`) |
| `preceding_runtime_contracts_present` | true (`e6e33dd`) |
| `preceding_validation_harness_present` | true (`73af6f6`) |
| `preceding_ui_shell_spec_present` | true (`745a980`) |
| `data_infra_spec_created` | true |
| `conceptual_store_map_created` | true |
| `ledger_audit_separation_defined` | true |
| `rollback_doctrine_defined` | true |
| `migration_gate_doctrine_defined` | true |
| `public_report_data_doctrine_defined` | true |
| `future_data_implementation_gates_defined` | true |
| `runtime_db_row_blind` | true |
| `runtime_db_accessed` | false |
| `schema_changed` | false |
| `migration_created` | false |
| `migration_executed` | false |
| `persistence_code_changed` | false |
| `runtime_behavior_changed` | false |
| `v1_release_preserved` | true |
| `cue_expansion_recommended` | false |
| Next recommended block | **`V2-EXTRACTION-SPEC-01` OR `V2-ROADMAP-02`** |

## Doctrine

Cue expansion remains explicitly **NOT** recommended. Private adapter
implementation remains blocked. V1 frozen release at `7ef8ffd`
continues to be the durable shipped artifact.

## Progress

- Whole MedAI project: **~96.6%** done / ~3.4% remaining.
