# B07-TERM Opt-In Integration Plan

Block: `B07-TERM-PLAN-01`

Status: design only. Runtime B07 behavior is unchanged.

## Current PARK-08 State

- TERM-05 synthetic read-only terminology adapter is ready.
- TERM-06 private-store read-only adapter validation is ready.
- TERM-07 UI-only terminology lookup panel is ready and hidden by default.
- TERM-08 hypothesis-only coding annotation pilot is ready and feature-flagged off by default.
- Local terminology capability covers RxNorm and LOINC only.
- The local terminology store remains private and uncommitted.
- No B07 integration is active.
- No accepted clinical facts are created by terminology lookup.
- No hypothesis promotion occurs.
- No DDI status clearing occurs.
- No external APIs are used.
- OCR, extraction, confidence, and safety gates are unchanged.

## Explicit Approval Boundary

B07-TERM implementation requires a separate explicit approval. This planning block is not approval to modify B07 runtime behavior.

## Required Feature Flags

- `MEDAI_B07_TERMINOLOGY_OPT_IN=false` by default.
- `MEDAI_TERMINOLOGY_LOOKUP_ENABLED=false` by default.
- `MEDAI_TERMINOLOGY_HYPOTHESIS_ANNOTATION=false` by default.
- `MEDAI_TERMINOLOGY_READ_ONLY=true`.
- `MEDAI_TERMINOLOGY_ALLOW_WRITES=false`.

All future implementation must fail closed when flags are missing or inconsistent.

## Proposed B07 Integration Points

Future B07 may read TERM-08 annotation objects only after all feature flags are explicitly enabled. The allowed integration point is a review-only metadata attachment on an already reviewed coding candidate.

Allowed data from TERM-08:

- normalized term status: `exact`, `ambiguous`, or `unmapped`
- source system
- candidate code count
- candidate code metadata
- reason codes
- review-required flag
- read-only lookup flag

## Forbidden B07 Integration Points

Future B07 must not:

- create accepted clinical facts from terminology lookup
- promote a hypothesis to active
- clear or downgrade DDI status
- generate clinical advice
- generate dosing or prescribing advice
- silently resolve ambiguity
- invent codes for unknown terms
- alter OCR, extraction, routing, confidence, or safety gates
- call external APIs
- treat terminology lookup as an authority source

## TERM-08 Input And Output Contract

Input: one candidate term string and optional source filter.

Output: a hypothesis-only annotation with:

- terminology status
- source system when unambiguous
- candidate codes
- `annotation_tier=hypothesis`
- `requires_review=true`
- reason codes
- `read_only_lookup=true`
- `writes_active_fact=false`
- `clears_ddi_status=false`
- `promotes_hypothesis=false`
- `external_api_used=false`

## Unknown And Ambiguous Behavior

Unknown terms must remain unmapped with no code hallucination.

Ambiguous terms must remain ambiguous and require manual review. There is no silent winner.

## Validation Matrix For Future Implementation

- Feature flags default off preserve baseline behavior.
- Enabled flags produce only hypothesis metadata.
- Unknown lookup remains unmapped.
- Ambiguous lookup remains manual-review.
- No active write occurs.
- No DDI status clearing occurs.
- No hypothesis promotion occurs.
- No external API call occurs.
- B07 tests pass unchanged with flags off.
- Public reports contain no private store details or source row content.

## Public Report Rules

Public reports may include aggregate counts, status labels, reason codes, and short commit identifiers. They must not include private store locations, license text, source rows, PHI, secrets, or runtime database details.

