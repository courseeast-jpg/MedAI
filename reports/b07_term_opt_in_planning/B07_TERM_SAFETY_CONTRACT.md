# B07-TERM Safety Contract

This contract applies to any future B07-TERM implementation.

## Invariants

- Feature flags are default off.
- Terminology lookup remains read-only.
- Terminology lookup is metadata support, not a clinical authority.
- Annotation tier remains `hypothesis`.
- Review remains required.
- Unknown terms remain unmapped.
- Ambiguous terms remain ambiguous and require manual review.
- No code hallucination is allowed.
- No accepted clinical fact is created by terminology lookup.
- No hypothesis promotion is allowed.
- No DDI status clearing or downgrading is allowed.
- No medication dosing or prescribing advice is allowed.
- No clinical recommendation generation is allowed.
- No external API is allowed.
- OCR, extraction, confidence, routing, and safety gates remain unchanged.

## Stop Conditions

Stop future implementation immediately if any test or report shows:

- active fact creation
- hypothesis promotion
- DDI status clearing
- ambiguity silently resolved
- unknown term mapped to invented code
- external API usage
- private terminology artifact staged
- public report privacy issue
- B07 baseline test regression with flags off

