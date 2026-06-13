# MEDAI Vertex Real Document Single Pilot Tokenization Spec 16C-B

## Status

This spec is no-live and synthetic only. No provider call, no Vertex live execution,
and no billing API call occur. 16D is not started.

## Token Scheme

- Each detected synthetic identifier is replaced with a stable token of the form
  `[CLASS_n]` (for example `[PATIENT_NAME_1]`).
- Repeated identical values map deterministically to the same token.
- Tokens are reversible only via the local token map, which is never sent and never
  reported.

## Covered Classes

Patient names, DOB, dates, addresses, phone numbers, email addresses, MRN, insurance
IDs, account IDs, provider names, facility names, specimen/accession IDs, filenames,
local paths, embedded metadata, OCR artifacts, free-text identifiers, and rare
combinations.

## Boundaries

No real/private document processing; no private corpus read; no corpus processing; no
PDF/image/OCR processing; no MKB write; no auto-accept; no medical decision; no
production queue mutation.
