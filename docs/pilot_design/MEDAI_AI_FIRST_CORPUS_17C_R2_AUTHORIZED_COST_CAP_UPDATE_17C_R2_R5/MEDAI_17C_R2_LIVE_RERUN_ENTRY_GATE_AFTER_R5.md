# MEDAI 17C-R2 Live Rerun Entry Gate After R5

17C-R2 live extraction is NOT re-run by this block.

## 17C-R2 May Be Re-Run Only When ALL Of The Following Hold

1. The authorized total cap is $0.40 and the per-chunk cap is $0.05 (this block).
2. The latest estimated total cost ($0.325362) is within the $0.40 cap.
3. The canonical 478 batch exists in the operator context and the sealed batch is valid
   (SHA256 + 478/478/478/0 + 0 residual PI).
4. The Vertex credential preflight reports PASS (project = sot-knowledge-ocr).
5. Explicit user authorization for the live run remains in force.
6. The run stays chunked (25), within the caps, with stop-on-first-failure and the live
   gate set only inside each chunk; no MKB write, no auto-accept, no medical decision.

## Recommended Operating Order

Because an external writer can delete the private batch between blocks, run the R4
operator-context restore immediately before the live run, then run 17C-R2 once. The
live runner re-verifies the batch at preflight.

## Hard Stops

If the estimate exceeds the cap, the canonical batch is missing/invalid, or credentials
are not ready, no provider call may occur. MKB import remains not started.
