# MEDAI-V2-ARCHITECTURE-SPEC-01 — Short Summary

Reports-only v2 architecture SPEC. Defines the v2 direction without
implementing anything. Preserves the frozen v1 local operator release.

## Decision

| Field | Value |
| --- | --- |
| `v1_release_preserved` | **true** |
| `v2_architecture_layers_defined` | **true** (10 layers A–J) |
| `v2_workstreams_defined` | **true** (9 workstreams) |
| `v2_non_goals_defined` | **true** |
| Top recommended next block | **`V2-FOUNDATION-SPEC-02`** |
| Recommended next 3-block sequence | `V2-FOUNDATION-SPEC-02` → `V2-RUNTIME-CONTRACTS-01` → `V2-VALIDATION-HARNESS-01` |
| `cue_expansion_recommended` | **false** |

## State

- Phase ID: `MEDAI-V2-ARCHITECTURE-SPEC-01`
- Mode: `v2_architecture_reports_only_spec`
- Branch: `clinical-knowledge-architecture`
- HEAD before this block: `1bf86dd`
- Freeze commit: `7ef8ffd`

## V2 architecture layers (10)

| ID | Layer |
| :-: | --- |
| A | Ingestion and source handling |
| B | Extraction / OCR |
| C | Document classification |
| D | Structured medical extraction (future work only) |
| E | Clinical knowledge / terminology (parked) |
| F | Safety and decision |
| G | Operator UI |
| H | Runtime data and persistence |
| I | Observability and validation |
| J | Packaging / deployment |

## V2 workstreams (9)

| ID | Risk |
| --- | :-: |
| `V2-FOUNDATION-SPEC-02` | low |
| `V2-RUNTIME-CONTRACTS-01` | low-moderate |
| `V2-VALIDATION-HARNESS-01` | low |
| `V2-UI-SHELL-SPEC-01` | low-moderate |
| `V2-DATA-INFRA-SPEC-01` | moderate |
| `V2-EXTRACTION-SPEC-01` | moderate |
| `V2-TERMINOLOGY-WAIT-GATE` | minimal |
| `V2-PACKAGING-SPEC-01` | low |
| `V2-ROADMAP-02` | minimal |

## Blocked / deferred (carried forward)

- Private adapter implementation, real private-store access
- Licensed terminology row reads, license-ack contents access
- External terminology APIs at runtime
- Cue expansion (explicitly **NOT** recommended)
- Autonomous diagnosis / treatment / medication / dose / DDI inference
- Real-corpus validation without a privacy-gated SPEC
- `MORE-UNKNOWN-DIAGNOSTICS`
- Runtime DB migrations without SPEC + rollback plan
- Production deployment automation
- Public-report row dumps under any condition

## Safety / privacy

- All standard invariants false.
- All 10 pre-existing tag groups (PARK-20..26, FREEZE, helper/wiring
  PARK-01, license-gate PARK-02) unchanged on origin.
- All 3 public reports privacy-clean.

## Progress

- Whole MedAI project: **~96.1%** done / ~3.9% remaining.
