# MEDAI Vertex Real Document Single Pilot 16D Entry Criteria 16C-C

This block is no-live. 16D is not started. The dedicated live gate
`MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_LIVE_APPROVED` must remain unset/inactive after 16C-C.

## 16D May Be Entered Only When ALL Of The Following Hold

1. Explicit user authorization and explicit operator approval are recorded.
2. Cost cap is confirmed.
3. Redaction/tokenization proof has passed (no raw PII/PHI, no token map in outbound).
4. One-document limit and one-call limit are enforced.
5. Stop-on-first-failure handling is in place.
6. Rollback/failure plan and evidence capture (before and after) are ready.
7. The dedicated live gate is set only inside the authorized 16D block.

## 16D Must Stop Immediately If

The live gate, approval record, cost cap, redaction proof, or one-call limit is
missing. No provider call, no real/private document processing, no corpus processing,
no PDF/image/OCR processing, no MKB write, no auto-accept, and no medical decision may
occur outside an explicitly authorized 16D execution.
