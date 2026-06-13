# MEDAI Vertex Real Document Single Pilot Design 16A

## Purpose

This is a no-live design package for a future single redacted-real-like or real-document Vertex pilot. It defines the criteria for a future block but does not execute that pilot.

## Pilot Scope

- Explicit no-live status: this block is design-only.
- Real-document live execution is not authorized.
- Required future live gate name placeholder: `MEDAI_FUTURE_SINGLE_DOC_VERTEX_LIVE_GATE_PLACEHOLDER`.
- One-document maximum.
- One-call limit.
- Redacted/tokenized payload only.
- No raw PII.
- No token map outbound; the token map remains local only.
- No active MKB write.
- No auto-accept.
- Review-required for every output.
- No medical decision output.
- Medication safety non-bypass is required if medication facts appear.
- Stop-on-first-failure rule.
- Bounded cost/token ceiling: future design must specify a hard token ceiling and cost cap before any call.
- Evidence anchoring requirements: every extracted fact must cite a source span from the redacted/tokenized payload.
- Declared label alias policy: only declared aliases may map source labels to normalized labels.
- Request-shape requirements: request body contains only `contents` and `generationConfig`; forbidden provider/MedAI metadata remains local.
- Report sanitization requirements: no raw private values, raw OCR/private payloads, raw PDF/image payloads, token maps, credentials, auth headers, or local secret paths.
- Operator approval requirements: named operator, timestamp, provenance declaration, billing/cost acknowledgement, and review-bound acknowledgement.

## Refusal Conditions

Refuse the future pilot if any required approval is missing, provenance is unknown, raw PII remains, token map would leave local custody, request shape is invalid, cost cap is missing, medication safety proof is missing when required, active write is requested, auto-accept is requested, medical advice is requested, or a previous failure has occurred.
