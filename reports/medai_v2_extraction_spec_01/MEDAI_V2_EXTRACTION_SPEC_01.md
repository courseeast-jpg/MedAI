# MEDAI-V2-EXTRACTION-SPEC-01

## Executive Summary

This block creates a reports-only V2 extraction and OCR architecture specification. It defines future source intake, text visibility, OCR routing, extraction adapter, fallback, parser, multilingual/script, review queue, observability, and rollback boundaries before any implementation occurs.

No extraction, OCR, classifier, threshold, cue, parser, fallback, DB, UI, runtime, terminology, or adapter behavior changed.

## Scope

In scope:

- Architecture-only extraction and OCR boundaries.
- Future implementation gates.
- Review-bound and local-only defaults.
- Aggregate-only observability doctrine.
- V1 frozen release preservation.

Out of scope:

- OCR routing changes.
- Extraction behavior changes.
- Classifier, threshold, cue, parser, fallback, UI, DB, runtime, or adapter implementation.
- Source/private document access.
- External API use.
- Terminology/private adapter implementation.

## Prior V2 Dependency Chain

- MEDAI-V2-ARCHITECTURE-SPEC-01
- MEDAI-V2-FOUNDATION-SPEC-02
- MEDAI-V2-RUNTIME-CONTRACTS-01
- MEDAI-V2-VALIDATION-HARNESS-01
- MEDAI-V2-UI-SHELL-SPEC-01
- MEDAI-V2-DATA-INFRA-SPEC-01

## Source Intake Boundary

Future ingestion-to-extraction handoff must pass structured request objects across the seam. Raw source content remains private operational data and must not appear in public reports. This block reads no source documents and implements no intake code.

## Text Visibility Boundary

Future text visibility categories:

- `text_layer_sufficient`
- `text_layer_insufficient`
- `table_structure_visible_text_insufficient`
- `image_like_pdf`
- `OCR_required_review`
- `extraction_error`
- `unknown_visibility`

All categories remain review-bound by default. This block implements no detector logic.

## OCR Routing Boundary

OCR routing must be an isolated adapter decision. No OCR route may become default-on without a guarded implementation block, synthetic tests, aggregate-only reports, rollback, and local-only posture unless separately approved. No OCR is run in this block.

## Extraction Adapter Boundary

Future adapter seams aligned with V2 runtime contracts:

- Local deterministic text extractor
- Table/layout extractor
- Local OCR extractor
- Classifier pre-pass
- Structured medical parser
- Review queue emitter

No concrete adapters are implemented.

## Confidence And Fallback Isolation

Future fallback rules:

- Fallback attempts must be traceable.
- Fallback result cannot silently overwrite higher-quality deterministic output.
- Empty fallback cannot become terminal over non-empty local result without an explicit rule.
- Confidence downgrade must be visible.
- Review-band status must remain visible.

This block changes no current fallback behavior.

## Structured Parser Boundary

Future parser behavior remains contract-only. No clinical interpretation expansion, diagnosis/treatment inference, medication/dose inference expansion, DDI behavior change, licensed terminology mapping, or cue expansion is allowed in this block. Parser output remains review-bound unless a later safety spec approves otherwise.

## Multilingual And Script Boundary

Future English, Russian, and Cyrillic handling is architecture-only here. No OCR, language detector, cue pack, or threshold change is implemented. Ambiguous language/script cases remain review-bound.

## Review Queue Boundary

Future extraction outputs should feed review queue contracts with controlled-vocabulary operator-visible reasons. No auto-accept is introduced. Raw text and filenames must not appear in public reports.

## Observability And Audit Boundary

Future extraction observability must be aggregate-only:

- Counts by visibility category
- Counts by route family
- Counts by fallback family
- Counts by review reason
- Booleans for external API use, source document access, and raw text output

No raw payload dumps are allowed.

## Rollback And Parking Boundary

Future extraction behavior changes require a pre-change baseline, focused synthetic fixtures, aggregate-only replay, rollback switch or revert path, staged safety check, and parking/freeze preservation. Existing parked tracks remain parked unless explicitly reopened.

## Future Implementation Gates

A. OCR routing changes require a separate guarded implementation block.  
B. Extraction adapter implementation requires preceding contract-conformance checks.  
C. Threshold/scoring changes require a safety spec and synthetic validation.  
D. Classifier behavior changes require review-bound aggregate replay.  
E. Parser behavior changes require clinical safety review.  
F. Fallback behavior changes require empty/non-empty fallback regression tests.  
G. Language/script detector changes require multilingual synthetic fixtures.  
H. Table/layout extraction changes require aggregate-only replay and no raw table dumps.  
I. External extraction APIs require privacy/safety spec and remain blocked by default.  
J. Cue pack expansion remains not recommended and out of scope.

## Validation Matrix

This spec must be validated by focused report tests, the extraction-spec audit script, prior V2 spec tests, privacy checks, final CKA MVP validation, B07 validation, ROUTE-FIX validation, UI ops validation, UI boot validation, and staged safety checks.

## Recommended Next Block

`V2-ROADMAP-02`

V2 extraction implementation must not start in this block. OCR routing, extraction behavior, thresholds, classifier behavior, cue packs, terminology/private adapter implementation, and cue expansion stay unchanged. The frozen V1 release baseline remains preserved.
