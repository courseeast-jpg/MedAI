# MEDAI-REVIEW-01 B07 Terminology Metadata Operator Review

This package is for operator review of B07-TERM-01 behavior. It is documentation and validation evidence only.

## What B07-TERM-01 Does

B07-TERM-01 adds an opt-in helper that can attach terminology lookup output as review metadata. The metadata is hypothesis-only and requires review.

When explicitly enabled, the helper can report:

- terminology status: `exact`, `ambiguous`, or `unmapped`
- source system when unambiguous
- candidate code count
- safe candidate code metadata
- reason codes
- `annotation_tier=hypothesis`
- `requires_review=true`
- `read_only_lookup=true`
- `b07_authority_source=false`

## What It Does Not Do

B07-TERM-01 does not create accepted clinical facts, promote hypotheses, clear DDI status, generate clinical advice, generate dosing or prescribing advice, change OCR, change extraction, change routing, change confidence thresholds, or change safety gates.

Terminology lookup is not an authority source.

## Feature Flags

Safe default values:

- `MEDAI_B07_TERMINOLOGY_OPT_IN=false`
- `MEDAI_TERMINOLOGY_LOOKUP_ENABLED=false`
- `MEDAI_TERMINOLOGY_HYPOTHESIS_ANNOTATION=false`
- `MEDAI_TERMINOLOGY_READ_ONLY=true`
- `MEDAI_TERMINOLOGY_ALLOW_WRITES=false`

All required opt-in flags must be enabled, read-only must remain true, and writes must remain false before metadata is produced. Missing or inconsistent flags fail closed.

## Default-Off Behavior

With default flags, B07-TERM produces no terminology metadata effect. This is the rollback state and preserves existing B07 behavior.

## Hypothesis-Only Metadata

Hypothesis-only means the metadata may help a reviewer inspect a candidate, but it is not a clinical fact and is not a decision. Operator review remains required.

## Unknown And Ambiguous Results

`unmapped` means the term was not matched and no code was invented.

`ambiguous` means more than one possible mapping exists. It remains manual-review and no silent winner is selected.

## Operator Review Checklist

Before any future expansion, verify:

- flags-off behavior remains unchanged
- opt-in metadata remains review-only
- unknown terms remain unmapped
- ambiguous terms remain manual-review
- no accepted facts are created
- no hypothesis promotion occurs
- no DDI status is cleared
- no clinical, dosing, or prescribing advice is generated
- public reports remain privacy-clean

## Rollback Instructions

Return all feature flags to the safe default values listed above. That disables B07 terminology metadata and restores no-effect behavior.

