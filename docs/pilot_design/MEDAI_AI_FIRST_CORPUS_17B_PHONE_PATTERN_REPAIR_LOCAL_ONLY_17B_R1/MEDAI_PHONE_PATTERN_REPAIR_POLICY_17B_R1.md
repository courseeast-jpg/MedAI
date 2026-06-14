# MEDAI Phone-Pattern Repair Policy 17B-R1

17B-R1 is local-only and does not call any AI provider. The validator is not weakened;
payloads are repaired first, then validation is rerun.

## Tokenization Rule

- Every residual 10-digit-style phone-like sequence that triggered 17B validation is
  replaced with a token `[PHONE_PATTERN_1]`, `[PHONE_PATTERN_2]`, and so on.
- If the surrounding context suggests barcode/specimen/order/accession rather than
  phone, the sequence is tokenized as `[NUMERIC_IDENTIFIER_1]`,
  `[NUMERIC_IDENTIFIER_2]`, and so on.
- No flagged 10-digit sequence is preserved raw in the outbound payload.

## Safety Posture

- Conservative by design: when in doubt, the sequence is tokenized.
- The validator rules are unchanged from 17B; only the payloads are repaired.
- The flagged numeric values are never written to public reports or to Downloads.

## Re-Validation

After repair, all 12 outbound requests are re-validated with the 17B rules. The block
reports PASS only when there are zero residual phone-pattern failures across all 12.
If any failure remains, the block reports BLOCKED and no provider call occurs.
