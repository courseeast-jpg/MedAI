# MEDAI-V2-RUNTIME-CONTRACTS-01 — Short Summary

Reports-only / typing-only v2 runtime interface contracts. 32 contracts
across 10 boundaries. No concrete adapters. No runtime wiring. No
runtime behavior change. V1 frozen release preserved.

## Deliverables

- `clinical_knowledge/v2_contracts/__init__.py` — package initializer.
- `clinical_knowledge/v2_contracts/runtime_contracts.py` — typing-only
  contracts (standard-library only, side-effect-free).
- `scripts/run_medai_v2_runtime_contracts_01.py` — reports-only audit.
- `tests/test_medai_v2_runtime_contracts_01.py` — focused tests.
- 3 public-safe reports.

## Contract inventory (10 boundaries × 32 contracts)

| Boundary | Contracts |
| --- | --- |
| Ingestion | `V2SourceKind`, `V2DocumentSource`, `V2IngestionRequest`, `V2IngestionAdapterProtocol` |
| Text visibility / document quality | `V2VisibilityStatus`, `V2TextVisibilityProfile`, `V2DocumentQualityProtocol` |
| Extraction / OCR orchestration | `V2ExtractionMode`, `V2ExtractionRequest`, `V2ExtractionResult`, `V2ExtractionAdapterProtocol` |
| Document classification | `V2DocumentTypeCandidate`, `V2ClassificationResult`, `V2ClassifierProtocol` |
| Clinical knowledge / terminology | `V2TerminologyQuery`, `V2TerminologyMatchSummary`, `V2TerminologyLookupProtocol` |
| Review / HITL | `V2ReviewDisposition`, `V2ReviewItem`, `V2ReviewQueueProtocol` |
| Operator action | `V2OperatorActionKind`, `V2OperatorActionRequest`, `V2OperatorActionResult`, `V2OperatorActionProtocol` |
| Audit / observability | `V2AuditEventKind`, `V2AuditEvent`, `V2ObservabilitySinkProtocol` |
| Validation report | `V2ValidationStatus`, `V2ValidationReceipt`, `V2ValidationHarnessProtocol` |
| Runtime privacy / safety | `V2RuntimeSafetyProfile`, `V2PrivacyGateProtocol` |

## Default `V2RuntimeSafetyProfile`

| Field | Default |
| --- | :-: |
| `local_only` | true |
| `review_bound` | true |
| `external_api_blocked` | true |
| `auto_accept_allowed` | false |
| `terminology_lookup_aggregate_only` | true |
| `private_adapter_implemented` | false |
| `cue_expansion_recommended` | false |
| `clinical_decision_expansion` | false |

## Verdict

| Field | Value |
| --- | --- |
| `block_mode` | reports_only_typing_only |
| `requires_preceding_spec` | true |
| `preceding_architecture_spec_present` | true (`551af98`) |
| `preceding_foundation_spec_present` | true (`8b53d82`) |
| `contracts_module_created` | true |
| `contracts_importable` | true |
| `contracts_side_effect_free` | true |
| `standard_library_only` | true |
| `concrete_adapters_implemented` | false |
| `runtime_wiring_added` | false |
| `runtime_behavior_changed` | false |
| `v1_release_preserved` | true |
| `cue_expansion_recommended` | false |
| Next recommended block | **`V2-VALIDATION-HARNESS-01`** |

## Progress

- Whole MedAI project: **~96.3%** done / ~3.7% remaining.
