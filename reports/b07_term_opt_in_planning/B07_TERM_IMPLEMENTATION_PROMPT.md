# Future Implementation Prompt

Use this prompt only after explicit approval.

Start B07-TERM-01 opt-in hypothesis-only terminology metadata integration.

Use TERM-08 annotation output as review-only metadata. Keep all feature flags default off:

- `MEDAI_B07_TERMINOLOGY_OPT_IN=false`
- `MEDAI_TERMINOLOGY_LOOKUP_ENABLED=false`
- `MEDAI_TERMINOLOGY_HYPOTHESIS_ANNOTATION=false`
- `MEDAI_TERMINOLOGY_READ_ONLY=true`
- `MEDAI_TERMINOLOGY_ALLOW_WRITES=false`

Implement no accepted fact creation, no hypothesis promotion, no DDI status clearing, no dosing advice, no prescribing advice, no clinical recommendation generation, no external API use, and no silent ambiguity resolution. Unknown terms must remain unmapped. Ambiguous terms must remain manual-review.

Add tests proving flags-off behavior is identical to baseline and flags-on behavior creates only hypothesis metadata. Keep private terminology artifacts and runtime DB/index files uncommitted. Stop on any safety regression.
