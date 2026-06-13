# MEDAI Vertex Real Document Single Pilot Go/No-Go Checklist 16A

## GO Requirements

- Human authorization required.
- Cost cap acknowledgement required.
- Dedicated future live gate required: `MEDAI_FUTURE_SINGLE_DOC_VERTEX_LIVE_GATE_PLACEHOLDER`.
- One-call limit required.
- One-document limit required.
- Redacted/tokenized only.
- No raw reports.
- No active writes.
- No auto-accept.
- No medical decision logic.
- Medication safety proof required if medication facts appear.
- Review-required output handling.

## NO-GO Blockers

- Missing human authorization.
- Missing billing/cost acknowledgement.
- Missing dedicated future live gate.
- More than one document.
- More than one call.
- Raw PII or raw private content remains.
- Token map outbound would occur.
- Active write requested.
- Auto-accept requested.
- Medical decision logic requested.
- Medication safety proof missing when medication facts appear.
- Request shape contains forbidden metadata.
- Report sanitization fails.
