# MEDAI Vertex Real Document Single Pilot Test Plan 16A

## Pre-Live No-Live Validations

- Run the 15Z release snapshot validation.
- Run PII stripping proof.
- Run vault isolation proof.
- Run adapter dry-run request shape validation.
- Run authorization and cost refusal checks.
- Run medication safety non-bypass checks.
- Run integrated readiness harness.
- Run operator UAT.

## Future Live Pilot Execution Outline

This document intentionally does not include commands that set a live gate. A separate future block must define any live execution procedure.

## Expected Pass Criteria

- Exactly one document selected.
- Exactly one call planned.
- Redacted/tokenized payload only.
- Evidence anchors present for every candidate fact.
- Declared label aliases only.
- Review-required output.
- No active write, no auto-accept, and no medical decision output.

## Expected Fail Criteria

- Any raw private value remains.
- Token map would be sent outbound.
- Request shape includes forbidden metadata.
- Provider or billing error occurs.
- Evidence anchoring fails.
- Medication safety proof is missing when required.
- Any active write, auto-accept, or medical decision output is requested.

## Stop Conditions

Stop on first failure, including privacy, request-shape, cost, provider, billing, evidence-anchor, medication-safety, review-boundary, active-write, auto-accept, or medical-decision failures.

## Rollback / Cleanup Expectations

No active MKB records or production review queue mutations should exist. Generated future reports must be sanitized and review-bound.

## Report Requirements

Reports must include no-live validation provenance, one-call and one-document proof, sanitized request fingerprint, evidence-anchor status, review-required status, cost cap status, and privacy result.
