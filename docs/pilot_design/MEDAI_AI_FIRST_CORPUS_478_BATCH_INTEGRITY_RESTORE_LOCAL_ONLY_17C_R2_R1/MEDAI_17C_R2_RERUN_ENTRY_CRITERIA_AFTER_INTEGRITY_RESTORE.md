# MEDAI 17C-R2 Rerun Entry Criteria After Integrity Restore

17C-R2 live extraction is NOT re-run by this block.

## 17C-R2 May Be Re-Run Only When ALL Of The Following Hold

1. This integrity restore reported PASS: 478 lines / 478 parseable / 478 unique / 0
   malformed, request_validation_passed, 0 residual PI failures, with sidecars sealed.
2. The live runner verifies the canonical JSONL SHA256 matches the sealed sidecar and
   the counts match before any provider call.
3. The Vertex credential preflight reports PASS (working ADC for sot-knowledge-ocr; no
   interactive login pending).
4. Explicit user authorization for the live run remains in force.
5. The run stays chunked (25), within $0.05/chunk and $0.25 total caps, with
   stop-on-first-failure and the live gate set only inside each chunk.

## Hard Stops

If integrity verification fails, or credentials are not ready, or authorization is
absent, no provider call may occur. MKB import remains not started.
