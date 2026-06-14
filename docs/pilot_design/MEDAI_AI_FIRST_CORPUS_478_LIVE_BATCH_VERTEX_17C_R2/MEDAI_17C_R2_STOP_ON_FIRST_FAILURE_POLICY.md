# MEDAI 17C-R2 Stop-On-First-Failure Policy

The run stops immediately after the first occurrence of any of the following, with no
retry and no fallback:

- Credential/auth failure.
- Request batch missing or malformed.
- Wrong request count (not exactly 478).
- Privacy validation failure (residual raw PI in a request).
- Cost cap breach (per-chunk > $0.05 or total > $0.25).
- Provider error.
- Quota / billing / auth error.
- Response parse failure.
- Schema validation failure.
- Raw PI leak detected in a response or public report.
- Live gate lifecycle failure.
- MKB DB open/write attempt.
- Unexpected original source file inclusion.
- Any request outside the canonical 478 batch.

On stop: the live gate is cleared in a finally block; successful private responses (if
any) are kept in private staging outside the repo; public reports carry counts and a
sanitized failure stage/category only; no MKB write, no auto-accept, no medical decision.
