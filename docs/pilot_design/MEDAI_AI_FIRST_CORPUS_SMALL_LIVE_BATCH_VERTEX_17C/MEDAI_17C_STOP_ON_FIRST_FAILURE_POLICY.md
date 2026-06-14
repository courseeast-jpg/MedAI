# MEDAI 17C Stop-On-First-Failure Policy

The 17C run stops immediately after the first occurrence of any of the following, with
no retry and no fallback:

- Privacy validation failure (residual raw PI in a request).
- Provider error (Vertex/Gemini error).
- Quota / billing / auth error.
- Response parse failure.
- Schema validation failure.
- Raw PI leak detected in a response or public report.
- Cost estimate breach (> $0.05).
- Live gate lifecycle failure.
- Unexpected request file count (not 12).
- Any MKB write attempt.

On stop:

- The live gate is cleared in a finally block.
- Successful private responses (if any) are kept in private staging, outside the repo.
- Public reports carry counts and a sanitized failure stage/category only.
- No MKB write, no auto-accept, and no medical decision occur.
