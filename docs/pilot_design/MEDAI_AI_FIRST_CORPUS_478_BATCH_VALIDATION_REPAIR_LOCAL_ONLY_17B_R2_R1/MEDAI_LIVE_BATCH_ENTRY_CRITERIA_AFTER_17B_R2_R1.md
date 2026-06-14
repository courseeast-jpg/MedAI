# MEDAI Live Batch Entry Criteria After 17B-R2-R1

Local-only. This block does not start live extraction and sets no live gate.

## Live Extraction May Be Considered Only When ALL Of The Following Hold

1. Explicit new user authorization for a corpus live extraction is granted.
2. 17B-R2-R1 reported `request_validation_passed=true` with
   `request_validation_failed_count_after_repair=0` across all 478 requests.
3. The Vertex credential preflight reports PASS (working ADC for project
   sot-knowledge-ocr; no interactive login pending).
4. A human/operator has reviewed the repair summary and validation report.
5. The run is executed in bounded chunks (suggested batch size ~25) with a hard cost cap
   per chunk and stop-on-first-failure.
6. The dedicated live gate is set only inside the authorized live flow and reset
   afterward.

## Hard Stops

If any request fails validation, or credentials are not ready, or authorization is
absent, no provider call may occur. No MKB write, no auto-accept, and no medical decision
occur here or as a side effect of preparing the live batch.
