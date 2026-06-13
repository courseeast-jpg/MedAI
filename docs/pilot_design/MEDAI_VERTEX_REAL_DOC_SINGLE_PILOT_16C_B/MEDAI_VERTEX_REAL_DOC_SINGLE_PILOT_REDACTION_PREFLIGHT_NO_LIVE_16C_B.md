# MEDAI Vertex Real Document Single Pilot Redaction Preflight No-Live 16C-B

## Status

- This block is no-live.
- 16C-B does not authorize live execution.
- 16C-B does not set the live gate.
- 16D is not started.
- All fixtures are synthetic only; no real/private document may be used.

## Hard Boundaries

This block enforces: no provider call; no Vertex live execution; no Gemini live
execution; no Claude/OpenAI live execution; no billing API call; no real/private
document processing; no private corpus read; no corpus processing; no PDF/image/OCR
processing; no MKB write; no auto-accept; no medical decision; no production queue
mutation.

## Redaction/Tokenization Preflight Requirement

A redaction/tokenization preflight must run before any future outbound payload is
assembled. The preflight must detect and block or tokenize at least: patient names,
DOB, dates, addresses, phone numbers, email addresses, MRN, insurance IDs, account
IDs, provider names, facility names, specimen/accession IDs, filenames and local
paths, embedded metadata, OCR artifacts, token maps, free-text identifiers, and rare
combinations that could re-identify a person.

## Pre-16D Requirements

- Explicit operator approval is required before any later 16D.
- Redaction/tokenization proof is required before any later 16D.
- Cost cap proof is required before any later 16D.
- One-document limit and one-call limit are required before any later 16D.
- Stop-on-first-failure is required before any later 16D.
- Rollback and failure plan are required before any later 16D.

## GCP Synthetic Sandbox Separation

The GCP synthetic sandbox evidence proves only environment, authentication, and the
request path. It is separate environment evidence only and does not prove MedAI
real-document readiness.
