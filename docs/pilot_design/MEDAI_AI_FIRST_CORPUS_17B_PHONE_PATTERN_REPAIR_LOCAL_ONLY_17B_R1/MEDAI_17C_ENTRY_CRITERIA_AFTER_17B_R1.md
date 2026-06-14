# MEDAI 17C Entry Criteria After 17B-R1

17B-R1 is local-only and does not start 17C. A future 17C small live batch requires
explicit new authorization and is not started here. The dedicated live gate remains
unset/inactive after 17B-R1.

## 17C May Be Considered Only When ALL Of The Following Hold

1. Explicit new user authorization for a 17C small live batch is granted.
2. 17B-R1 reported `request_validation_passed=true` with
   `residual_phone_pattern_failures_after_repair=0` across all 12 ready files.
3. A human/operator has reviewed the repair summary and the dry-run outputs.
4. A hard cost cap is confirmed for the batch.
5. The outbound requests contain tokens and tokenized clinical content only — no raw
   PI, no token map, no flagged raw numeric sequence.
6. A bounded batch size and per-call limit are set.
7. Stop-on-first-failure handling is in place.
8. The dedicated live gate is set only inside the authorized 17C flow and reset
   afterward.

## Hard Stops

If any outbound request fails validation, or authorization is absent, no provider call
may occur. No medical decision, no MKB write, and no auto-accept occur in 17B-R1 or as
a side effect of preparing 17C. The 587 blocked files remain excluded.
