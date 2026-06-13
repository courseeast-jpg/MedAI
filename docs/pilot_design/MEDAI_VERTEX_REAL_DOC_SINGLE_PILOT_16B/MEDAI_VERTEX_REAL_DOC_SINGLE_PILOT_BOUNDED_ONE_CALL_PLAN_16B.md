# MEDAI Vertex Real Document Single Pilot Bounded One-Call Plan 16B

## Bounded Execution Outline (Future, Not Run Here)

- Exactly one document selected.
- Exactly one call planned (one-call limit).
- Redacted/tokenized payload only; no raw PII; token map remains local only.
- Request shape contains only `contents` and `generationConfig`; no forbidden metadata.
- Bounded token ceiling and hard cost cap enforced before the call.
- Evidence anchoring: every candidate fact must cite a source span from the
  redacted/tokenized payload.
- Declared label alias policy: only declared aliases may map source labels.
- Review-required output; no active MKB write; no auto-accept; no medical decision.

## Explicitly Not Included

This document intentionally does not include any command that sets a live gate or makes
a provider call. A separate future block must define any live execution procedure.

## Expected Pass Criteria

- One document, one call, redacted/tokenized payload only, evidence anchors present,
  declared label aliases only, review-required output, no active write, no auto-accept,
  and no medical decision output.

## Expected Fail Criteria

- Any raw private value remains, token map would be sent outbound, request shape
  includes forbidden metadata, provider/billing error occurs, evidence anchoring fails,
  or medication safety proof is missing when required.
