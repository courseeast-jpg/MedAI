# MEDAI-RU-DOC-TYPE-02 Real Text Path Diagnostic

Conclusion: medai_ru_doc_type_02_completed

## Scope

Changed scope: Russian lab text-path diagnostic plus a narrow metadata-path fix.

The diagnostic found a metadata integration issue: an existing display metadata value of `unknown` or `generic` was treated as final before text-based document type classification could run. That could keep real Russian general lab documents labeled `Unknown` even when readable extracted text was available.

The fix keeps authoritative non-unknown labels unchanged, but lets available text classify the display label when the existing metadata is only an unknown fallback.

## Root Cause

Root cause: classifier input path was blocked by an unknown metadata value.

Evidence summary:
- Run & Review creates operator document type labels through `display_document_type`.
- The pipeline can pass existing metadata such as `unknown`.
- Before this block, `unknown` mapped directly to `Unknown` and returned without checking available extracted text.
- Synthetic runtime-path tests now prove `display_document_type("unknown", text=...)` can produce `Lab result` when multiple Russian lab cue categories are available.
- The latest public test-run report does not retain extracted text, so no raw source text was read or printed for this diagnostic.

## Safe Per-File Diagnostic Summary

Safe labels only:

| Safe ID | Document type before | Extractor | Confidence bucket | OCR/text quality | Text available in public report | Cyrillic detected | Cue categories detected | Likely reason unknown |
|---|---|---|---|---|---|---|---|---|
| file_001 | Unknown | spacy | moderate_0_50_to_0_64 | readable | false | false | none | classifier_input_field_missing |
| file_002 | Unknown | rules_based | low_under_0_50 | readable | false | false | none | classifier_input_field_missing |
| file_003 | Urinalysis | rules_based | low_under_0_50 | readable | false | false | none | unknown |

Note: the latest public test-run report intentionally excludes extracted text. Cue categories for the real source files were therefore not reconstructed from raw documents in this block.

## Behavior Changed

- Existing `unknown` or `generic` display metadata no longer blocks text-based metadata classification when text is available.
- Added privacy-safe diagnostic output for text availability, text length bucket, Cyrillic density bucket, cue-category names, and likely unknown reason.
- Added normalization for Cyrillic `ё` to `е`.

## Behavior Not Changed

- Urinalysis detection preserved.
- Treatment plan detection preserved.
- Medication plan detection preserved.
- Auto-acceptance changed: false.
- Confidence thresholds changed: false.
- Confidence scoring changed: false.
- Clinical logic changed: false.
- Clinical interpretation added: false.
- Medication advice added: false.
- DDI logic changed: false.
- OCR engine changed: false.
- External API enabled: false.
- Cloud API used: false.
- Extraction parser changed: false.
- Lab value parser added: false.
- Treatment parser added: false.
- Safety gate changed: false.
- B07 terminology changed: false.
- ROUTE-FIX changed: false.
- DB schema changed: false.
- Command behavior changed: false.
- Allowlist changed: false.
- Free-form shell added: false.

## Validation Summary

- `python -m pytest tests/test_medai_ru_doc_type_02_real_text_path_diagnostic.py -vv`: passed, 10 tests.
- `python -m pytest tests/test_medai_ru_doc_type_01_russian_detection.py -vv`: passed, 12 tests.
- `python -m pytest tests/test_medai_lab_fix_02_general_lab_detection.py -vv`: passed, 11 tests.
- `python -m pytest tests/test_medai_lab_fix_01_document_type_review_reason.py -vv`: passed, 9 tests.
- `python -m pytest tests/test_medai_ui_bugfix_02_readd_after_queue_clear.py -vv`: passed, 6 tests.
- `python -m pytest tests/test_medai_ui_bugfix_01_clear_report_upload_idempotency.py -vv`: passed, 6 tests.
- `python scripts/run_medai_ui_ops_panel_validation.py`: passed.
- `python scripts/run_medai_ui_boot_fix_validation.py`: passed.
- `python scripts/run_cka_final_mvp_release_validation.py`: passed, 12/12 cases, 693 tests reported.
- `python scripts/run_b07_term01_opt_in_integration_validation.py`: passed, 6/6 cases.
- `python scripts/run_medai_route_fix01_validation.py`: passed.
- `python -m pytest tests`: passed, 2366 passed, 4 skipped, 22 warnings.

## Privacy And Safety

- No raw PHI in report.
- No raw filenames in report.
- No raw document text in report.
- No private absolute paths in report.
- No secrets in report.
- No source documents staged.
- No test input files staged.
- No validation input files staged.
- No terminology files staged.
- No runtime DB files staged.
