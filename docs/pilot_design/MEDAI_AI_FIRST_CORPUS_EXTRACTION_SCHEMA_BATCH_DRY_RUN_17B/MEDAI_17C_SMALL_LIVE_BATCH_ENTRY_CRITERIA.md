# MEDAI 17C Small Live Batch Entry Criteria

17B is dry-run only and does not start 17C. A future 17C small live batch requires
explicit new authorization and is not started here. The dedicated live gate remains
unset/inactive after 17B.

## 17C May Be Considered Only When ALL Of The Following Hold

1. Explicit new user authorization for a 17C small live batch is granted.
2. The 17B dry run reported `request_validation_passed=true` for the 12 ready files
   (no residual raw PI in any outbound request).
3. A human/operator has reviewed the dry-run outputs and the prompt/schema contract.
4. Cost estimate is reviewed and a hard cost cap is confirmed.
5. The outbound requests contain tokens and tokenized clinical content only — no raw
   PI, no token map.
6. A bounded batch size and per-call limit are set for the small live batch.
7. Stop-on-first-failure handling is in place.
8. The dedicated live gate is set only inside the authorized 17C flow and reset
   afterward.

## Hard Stops

If any outbound request fails privacy validation, or the dry run is blocked, or
authorization is absent, no provider call may occur. No medical decision, no MKB write,
and no auto-accept occur in 17B or as a side effect of preparing 17C.
