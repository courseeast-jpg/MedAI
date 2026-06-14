# 17C-R2 rerun entry gate (after integrity restore)

- request_validation_passed_after: `True`
- rebuilt_request_count: `478` / `478`
- malformed_after: `0` | residual_pi_failures_after: `0`

17C-R2 live extraction is NOT re-run here. Before any rerun, verify the canonical
JSONL SHA256 matches the sealed sidecar and the counts match, confirm the Vertex
credential preflight PASSES, and keep the run chunked/capped with stop-on-first-failure.
