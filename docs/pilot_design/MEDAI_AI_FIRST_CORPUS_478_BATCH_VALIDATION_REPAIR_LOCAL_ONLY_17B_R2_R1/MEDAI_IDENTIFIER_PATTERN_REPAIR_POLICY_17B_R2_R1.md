# MEDAI Identifier Pattern Repair Policy 17B-R2-R1

Local-only. The validator is not weakened; payloads are repaired first, then re-validated
with the unchanged 17B-R2 rules.

## Failure Class Handling

1. phone -> residual phone-like numeric sequences become `[PHONE_PATTERN_n]`, or
   `[NUMERIC_IDENTIFIER_n]` when the context is non-phone but still identifier-like.
2. mrn -> residual MRN-like labels/values become `[MRN_n]`.
3. insurance_account -> insurance/policy/member or account/acct labels/values become
   `[INSURANCE_ID_n]` or `[ACCOUNT_ID_n]`.
4. accession_specimen -> accession/specimen/order/collection/barcode labels/values become
   `[ACCESSION_ID_n]`, `[SPECIMEN_ID_n]`, or `[ORDER_ID_n]`.

## Rules

- Repair is applied to a fixed point so that the validator's token-stripped view is free
  of all four identifier classes.
- Ambiguous-but-clinical numeric sequences are still tokenized; privacy is higher
  priority than preserving identifier utility for the first live pass.
- No flagged residual value is preserved raw in the outbound payload.
- Residual numeric values are never printed or committed.
