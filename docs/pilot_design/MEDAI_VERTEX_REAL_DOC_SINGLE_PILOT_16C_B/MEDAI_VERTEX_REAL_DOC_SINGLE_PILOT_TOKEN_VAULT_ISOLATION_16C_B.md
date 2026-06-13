# MEDAI Vertex Real Document Single Pilot Token Vault Isolation 16C-B

## Status

No-live, synthetic only. 16D is not started. No provider call and no billing API call
occur.

## Token Vault Rules

- Token maps must never be included in outbound payload.
- Token maps must never appear in public reports.
- Token maps must be isolated from any provider-facing payload.
- Any token-map leak is a hard NO-GO.
- Restoration of tokens is local-only and is not part of this block.

## Reporting

Public reports contain only safe summary counts, token labels, and fingerprints —
never raw synthetic identifiers and never token map values.

## Boundaries

No MKB write; no auto-accept; no medical decision; no production queue mutation.
