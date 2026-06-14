# MEDAI Local OCR De-Identification Review Rules 16E-A

## Status

Local-only. No provider call. No live gate activation. 16D retry is not started.

## Local Pipeline (No Network)

1. Read exactly the one approved image.
2. Run local OCR/extraction only (no network, no provider).
3. Run local detection/redaction/tokenization only.
4. Write raw artifacts privately, outside git.
5. Emit only sanitized counts/hashes/flags to public reports.

## Detection Classes Assessed

Patient name, DOB, dates, address, phone, email, MRN, insurance ID, account ID,
provider name, facility name, accession/specimen ID, filename/path, embedded
metadata, OCR artifacts, free-text identifiers, and rare re-identification
combinations.

## De-Identification Review Rules

- A label-anchored or pattern-based detector is treated as **incomplete** on
  unstructured real OCR text; it may under-detect.
- The certification cannot return PASS on automated detection alone.
- A residual scan looks for identifier-like patterns remaining after tokenization.
  Any residual candidate means the payload is not certified.
- Token maps must never appear in public reports and must never be sent.
- Public reports carry counts, hashes, and pass/fail flags only.

## Required Human Review

Because absence of identifiers cannot be proven automatically on real unstructured
text, human/operator review of the private tokenized payload is required before any
later live send. This block produces the private review material only; it does not
send anything.
