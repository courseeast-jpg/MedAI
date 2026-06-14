# MEDAI 16D Retry Entry Criteria After 16E-A

## Status

Local-only. This block does not authorize live execution and does not set the live
gate. 16D retry is not started. Future 16D retry requires explicit new authorization.

## A Future 16D Retry May Be Considered Only When ALL Of The Following Hold

1. Explicit new user authorization for a 16D retry is granted (separate from this block).
2. A human/operator has completed the No-PHI attestation (NO_PHI_ATTESTED) after a
   line-by-line review of the private tokenized payload.
3. The de-identification approach used has measured recall on real documents, not only
   synthetic fixtures.
4. The outbound payload is confirmed to contain tokens only, with no raw patient
   identifiers and no token map.
5. One-document limit and one-call limit remain enforced.
6. Stop-on-first-failure, rollback/failure plan, and evidence capture remain in place.
7. Cost cap remains confirmed.
8. The dedicated live gate is set only inside the authorized 16D retry flow and reset
   afterward.

## Hard Stops Carried Forward

If the attestation is NOT_ATTESTED, or any identifier may remain, or the token map
could leave local custody, no live send may proceed. No provider call, no corpus
processing, no MKB write, no auto-accept, and no medical decision occur outside an
explicitly authorized 16D retry.
