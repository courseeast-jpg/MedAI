# MEDAI Private Batch Integrity Seal Policy 17C-R2-R1

After the canonical 478 batch is rebuilt and atomically replaced, three private sidecars
seal its integrity (all outside the repo, never committed):

- `outbound_requests_integrity_private.json` — line count, parseable count, unique doc-id
  count, malformed count, request_validation_passed, residual_pi_failures, and the full
  SHA256.
- `outbound_requests_private.sha256` — the SHA256 of the JSONL bytes.
- `doc_id_manifest_private.json` — the sorted list of the 478 doc IDs.

## Future Live Verification (Required)

Before any future provider call, a live runner must re-verify the canonical JSONL against
these sidecars: 478 lines, 478 parseable, 478 unique doc IDs, 0 malformed, and a SHA256
that matches the sealed value. Any mismatch is a hard stop before any provider call.

The rebuilt JSONL is set read-only after writing when supported, to reduce accidental
mutation. A future rebuild clears read-only, replaces atomically, then re-seals.
