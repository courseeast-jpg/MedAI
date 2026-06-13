# MEDAI Vertex Real Document Single Pilot Authorization Prep 16B

## Purpose

This is a no-live authorization-prep package for a future single-document Vertex
pilot. It prepares the pilot's own approval, the dedicated future live gate, a
bounded one-call execution plan, and stop-on-first-failure handling. It does not run
the pilot.

## Status

- Explicit no-live status: this block is design-only.
- Real-document live execution is not authorized.
- This block builds on the frozen 15Z/16A governance state and does not modify it.
- The GCP synthetic sandbox remains separate environment evidence only and is not
  treated as MedAI real-doc readiness validation.

## Own Approval Requirement

This future pilot requires its own approval, separate from 16A. Human authorization required before any future live pilot. A named operator must complete the 16B approval record. There is no auto-accept of approval and no default authorization.

## Dedicated Future Live Gate

Required future live gate name placeholder: `MEDAI_FUTURE_SINGLE_DOC_VERTEX_LIVE_PILOT_GATE_PLACEHOLDER`. This block does
not set that gate and contains no command that sets a live gate. The gate must be
defined, separately authorized, and explicitly set only in a later, separately
approved execution block.

## Bounded One-Call Execution Plan

- One-document maximum.
- One-call limit.
- Redacted/tokenized payload only.
- No raw PII.
- No token map outbound; the token map remains local only.
- Bounded token ceiling and a hard cost cap are required before any call.
- Request shape: the request body contains only `contents` and `generationConfig`;
  forbidden provider/MedAI metadata remains local.
- Review-required for every output; no active MKB write; no auto-accept; no medical
  decision output.
- Medication safety non-bypass is required if medication facts appear.

## Stop-On-First-Failure Handling

Stop-on-first-failure rule applies. On the first privacy, request-shape, cost,
provider, billing, evidence-anchor, medication-safety, review-boundary, active-write,
auto-accept, or medical-decision failure, halt immediately, preserve sanitized
evidence, and require fresh re-authorization.

## Refusal Conditions

Refuse the future pilot if any required approval is missing, provenance is unknown,
raw PII remains, token map would leave local custody, request shape is invalid, cost
cap is missing, medication safety proof is missing when required, active write is
requested, auto-accept is requested, medical advice is requested, or a previous
failure has occurred.
