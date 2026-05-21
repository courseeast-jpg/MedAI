# MEDAI-V2-EXTRACTION-SPEC-01 Report

## Scope And Non-Scope

This is a reports-only V2 extraction/OCR architecture spec. It defines future boundaries and gates only.

It does not implement extraction, OCR, OCR routing, classifier changes, threshold/scoring changes, cue packs, parser behavior, fallback behavior, runtime wiring, adapters, UI, DB/schema/migration changes, terminology/private adapter work, external API use, or private data access.

## Prior V2 Dependency Chain

- MEDAI-V2-ARCHITECTURE-SPEC-01
- MEDAI-V2-FOUNDATION-SPEC-02
- MEDAI-V2-RUNTIME-CONTRACTS-01
- MEDAI-V2-VALIDATION-HARNESS-01
- MEDAI-V2-UI-SHELL-SPEC-01
- MEDAI-V2-DATA-INFRA-SPEC-01

## Source Intake Boundary

Future ingestion-to-extraction handoff must use structured request objects. Raw source content remains private operational data and is not permitted in public reports. No source documents were opened in this block.

## Text Visibility Boundary

Future text visibility categories are `text_layer_sufficient`, `text_layer_insufficient`, `table_structure_visible_text_insufficient`, `image_like_pdf`, `OCR_required_review`, `extraction_error`, and `unknown_visibility`. These categories remain review-bound by default.

## OCR Routing Boundary

OCR routing is a future isolated adapter decision. No OCR route may become default-on without a separate guarded implementation block, synthetic fixtures, aggregate-only reports, rollback, and local-only posture unless separately approved. No OCR ran in this block.

## Extraction Adapter Boundary

Future adapter seams are local deterministic text extractor, table/layout extractor, local OCR extractor, classifier pre-pass, structured medical parser, and review queue emitter. No concrete adapters were implemented.

## Confidence And Fallback Isolation

Fallback attempts must be traceable. Fallback output must not silently overwrite higher-quality deterministic output. Empty fallback must not become terminal over a non-empty local result without an explicit rule. Confidence downgrades and review-band status must remain visible.

## Structured Parser Boundary

Parser behavior remains contract-only. This block adds no clinical interpretation expansion, diagnosis/treatment inference, medication/dose inference expansion, DDI behavior change, licensed terminology mapping, or cue expansion.

## Multilingual And Script Boundary

English, Russian, and Cyrillic handling is architecture-only in this spec. No OCR, language detector, cue pack, or threshold change is implemented. Ambiguous language/script cases remain review-bound.

## Review Queue Boundary

Future extraction outputs must feed review queue contracts with controlled-vocabulary operator-visible reasons. No auto-accept is introduced. Raw text and filenames are prohibited from public reports.

## Observability And Audit Boundary

Future extraction observability must be aggregate-only: counts by visibility category, route family, fallback family, and review reason, plus booleans for external API use, source document access, and raw text output. Raw payload dumps are prohibited.

## Rollback And Parking Boundary

Any future extraction behavior change must include a pre-change validation baseline, focused synthetic fixtures, aggregate-only replay, rollback switch or revert path, staged safety check, and parking/freeze preservation. Existing parked tracks stay parked unless explicitly reopened.

## Safety And Privacy Invariants

- Extraction behavior changed: false
- OCR behavior changed: false
- OCR routing changed: false
- Classifier changed: false
- Threshold/scoring changed: false
- Cue pack changed: false
- Parser behavior changed: false
- Fallback behavior changed: false
- Runtime behavior changed: false
- UI changed: false
- DB/schema/migration changed: false
- Runtime wiring added: false
- External API used: false
- Private data accessed: false
- Source documents opened: false
- Raw text read or printed: false
- Raw filenames printed: false
- Private paths printed: false
- Secrets printed: false
- Licensed rows read or exposed: false
- Private license acknowledgement read: false
- Private config read: false
- Runtime DB accessed: false
- Tags touched: false
- Cue expansion recommended: false
- Review-bound default: true
- Local-only default: true
- External API blocked by default: true
- Auto-accept allowed by default: false
- V1 release preserved: true

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
J. Cue pack expansion remains explicitly not recommended and out of scope.

## Validation Matrix

Validation includes focused extraction spec tests, prior V2 data-infra/UI-shell/validation-harness/runtime-contract tests, extraction-spec audit script, public report privacy checks, final CKA MVP validation, B07 validation, ROUTE-FIX validation, UI ops validation, UI boot validation, and staged safety check.

## Recommended Next Block

`V2-ROADMAP-02`

V2 extraction implementation must not start in this block. The frozen V1 release baseline remains preserved.

## Validation Results

- Focused V2 extraction spec tests: passed, 7 tests.
- Prior V2 regression pack: passed, 90 tests across data-infra, UI shell, validation harness, and runtime contracts.
- Extraction-spec audit script: passed.
- Public report privacy checks: passed for all three V2 extraction spec reports.
- Final CKA MVP validation: passed, 12/12 cases and 693 tests, external API used false.
- B07 term01 validation: passed, 6/6 cases, external API used false.
- ROUTE-FIX validation: passed.
- UI ops validation: passed.
- UI boot validation: passed.
- Staged safety check: passed; only V2 extraction spec report, script, and test files were staged.
- Full pytest: not run; focused V2, prior V2, privacy, and V1 health validations cover this reports-only spec block.
