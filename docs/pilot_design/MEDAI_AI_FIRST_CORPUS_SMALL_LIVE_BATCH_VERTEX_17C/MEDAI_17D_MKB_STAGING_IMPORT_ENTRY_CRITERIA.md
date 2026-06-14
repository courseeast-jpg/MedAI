# MEDAI 17D MKB Staging Import Entry Criteria

17C does not write to MKB. A future 17D MKB staging import requires separate explicit
authorization and is not started by 17C.

## 17D May Be Considered Only When ALL Of The Following Hold

1. Explicit new user authorization for a 17D MKB staging import is granted.
2. 17C reported `execution_result=PASS` with all 12 responses schema-valid and zero PI
   leaks.
3. A human/operator has reviewed the private parsed responses and the schema validation.
4. The MKB import is a staging/dry-run first, review-bound, with no auto-accept and no
   medical decision.
5. Token restoration (if any) remains local-only and is never sent outbound.

## Hard Stops

If 17C did not pass, or authorization is absent, no MKB import occurs. No auto-accept
and no medical decision occur in 17C or as a side effect of preparing 17D.
