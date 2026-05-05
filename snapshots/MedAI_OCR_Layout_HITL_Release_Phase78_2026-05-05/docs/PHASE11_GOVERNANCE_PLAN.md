# Phase 11 Governance Plan

## Scope

Phase 11 adds a governance layer on top of the Phase 10 baseline without redesigning routing, extraction, or the core pipeline.

## Components

- `governance/truth_resolution.py`
  Deterministic conflict resolution overlay behind `ENABLE_TRUTH_RESOLUTION`.
- `governance/hypothesis_tier.py`
  Hypothesis-tier enforcement for AI- and web-derived records behind `ENABLE_HYPOTHESIS_TIER`.
- `governance/decision_scoring.py`
  Deterministic scoring wrapper behind `ENABLE_DECISION_SCORING`.
- `governance/governance_ledger.py`
  Append-only governance audit ledger.

## Adapter Hooks

- `app/config.py`
  Adds Phase 11 feature flags, all default `False`.
- `execution/pipeline.py`
  Uses governance truth-resolution adapter and hypothesis classifier. With flags disabled, behavior remains Phase 10-equivalent.

## Default Behavior

All Phase 11 components are feature-flagged off by default. The default runtime path must preserve Phase 10 behavior.

## Test Coverage

- Trust precedence rules
- Recency rule
- Multi-source replacement
- Numeric merge
- Medication-dose quarantine
- Fallback quarantine
- Hypothesis-tier enforcement
- Hypothesis exclusion from active context
- Default-off preservation of Phase 10 behavior
