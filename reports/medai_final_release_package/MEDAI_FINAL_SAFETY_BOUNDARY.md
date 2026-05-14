# MedAI Final Safety Boundary

## Clinical Boundary

MedAI is not a medical device and is not an autonomous diagnosis system.

MedAI does not provide medication dosing or prescribing advice as an authority.

MedAI output requires trained operator review for clinical use.

## Terminology Boundary

B07 terminology metadata is opt-in, default-off, hypothesis-only, and review-only.

Terminology lookup is not an authority source.

Unknown terms remain unmapped.

Ambiguous terms remain manual-review.

Terminology lookup does not clear or downgrade DDI status.

Terminology lookup does not promote hypotheses into accepted clinical facts.

## Runtime Boundary

External APIs remain off unless separately approved.

Private terminology source files, private license acknowledgment files, and runtime terminology stores remain uncommitted.

OCR, extractor, confidence threshold, routing, and safety gate boundaries remain as validated. This release package does not broaden those behaviors.

## Artifact Boundary

Public release artifacts must exclude private paths, source terminology rows, PHI, secrets, raw diffs, local runtime database paths, and vendor license excerpts.
