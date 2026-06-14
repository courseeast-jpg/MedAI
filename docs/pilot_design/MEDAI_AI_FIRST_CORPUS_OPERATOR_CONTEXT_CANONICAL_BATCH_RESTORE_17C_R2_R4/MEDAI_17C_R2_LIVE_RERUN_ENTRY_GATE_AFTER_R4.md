# MEDAI 17C-R2 Live Rerun Entry Gate After R4

17C-R2 live extraction is NOT re-run by this block.

## 17C-R2 May Be Re-Run Only When ALL Of The Following Hold

1. This block reported the canonical batch present in the operator context (exists,
   is_file, readable true) and the sealed batch valid: SHA256 matches; 478 physical
   newline records; 478 parseable; 478 unique doc IDs; 0 malformed; request validation
   passed; 0 residual PI failures.
2. The R3 resolver tests pass in the same operator context.
3. The Vertex credential preflight reports PASS (project = sot-knowledge-ocr).
4. Explicit user authorization for the live run remains in force.
5. The run stays chunked (25), within $0.05/chunk and $0.25 total caps, with
   stop-on-first-failure and the live gate set only inside each chunk.
6. The live runner re-verifies the canonical batch immediately before any provider call.

## Hard Stops

If the canonical batch is missing/invalid in the operator context, or credentials are
not ready, no provider call may occur. MKB import remains not started.
