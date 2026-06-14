# MEDAI 17C-R2 Live Rerun Entry Gate After R3

17C-R2 live extraction is NOT re-run by this block.

## 17C-R2 May Be Re-Run Only When ALL Of The Following Hold

1. This block reported the canonical batch resolved (exists/is_file/readable true) and
   the sealed batch valid: SHA256 matches the sidecar; 478 physical-newline records;
   478 parseable; 478 unique doc IDs; 0 malformed; request validation passed; 0 residual
   PI failures.
2. The 17C-R2 live runner uses the shared resolver and the physical-newline reader.
3. The Vertex credential preflight reports PASS (google.auth importable, ADC refresh
   succeeds, project = sot-knowledge-ocr).
4. Explicit user authorization for the live run remains in force.
5. The run stays chunked (25), within $0.05/chunk and $0.25 total caps, with
   stop-on-first-failure and the live gate set only inside each chunk.

## Hard Stops

If the canonical batch cannot be resolved, the sealed batch is invalid, or credentials
are not ready, no provider call may occur. MKB import remains not started.
