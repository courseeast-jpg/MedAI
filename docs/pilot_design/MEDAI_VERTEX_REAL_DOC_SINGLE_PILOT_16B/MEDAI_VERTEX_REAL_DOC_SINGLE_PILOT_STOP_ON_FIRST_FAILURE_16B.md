# MEDAI Vertex Real Document Single Pilot Stop-On-First-Failure 16B

## Rule

Stop-on-first-failure: a future pilot halts immediately on the first failure and makes
no further call.

## Trigger Conditions

Stop on first failure, including privacy, request-shape, cost, provider, billing,
evidence-anchor, medication-safety, review-boundary, active-write, auto-accept, or
medical-decision failures, or any uncertainty about payload privacy.

## On Trigger

- Stop immediately; retries are not permitted.
- Preserve sanitized evidence only (fingerprints, counts, token classes); never raw
  PII or token map.
- No active MKB write, no auto-accept, no medical decision, no production queue
  mutation.
- Require fresh explicit human authorization and a fresh approval before any retry.

## Rollback / Cleanup Expectations

No active MKB records or production review queue mutations should exist. Generated
future reports must be sanitized and review-bound.
