# MEDAI-RU-DOC-TYPE-02 Report

Conclusion: medai_ru_doc_type_02_completed

Changed scope: Russian lab text-path diagnostic and narrow metadata-path fix.

Root cause: existing `unknown` or `generic` document type metadata blocked text-based document type classification.

Evidence:
- Run & Review uses `display_document_type` for operator-facing labels.
- Existing unknown metadata was mapped directly to `Unknown`.
- Text cue classification was skipped in that path.
- The fix lets available text classify the display label only when existing metadata is an unknown fallback.
- The latest public test-run report excludes extracted text, so no raw source text was read or printed.

Safe per-file summary:

| Safe ID | Before | Extractor | Confidence bucket | OCR/text quality | Text available in public report | Cue categories | Likely reason unknown |
|---|---|---|---|---|---|---|---|
| file_001 | Unknown | spacy | moderate_0_50_to_0_64 | readable | false | none | classifier_input_field_missing |
| file_002 | Unknown | rules_based | low_under_0_50 | readable | false | none | classifier_input_field_missing |
| file_003 | Urinalysis | rules_based | low_under_0_50 | readable | false | none | unknown |

Outcome:
- Real Russian lab unknown issue explained: true
- Document type detection improved: true
- Russian lab result detection fixed for available cues: true
- Urinalysis detection preserved: true
- Treatment plan detection preserved: true
- Medication plan detection preserved: true

Safety boundary:
- Auto-acceptance changed: false
- Confidence thresholds changed: false
- Confidence scoring changed: false
- Clinical logic changed: false
- Clinical interpretation added: false
- Medication advice added: false
- DDI logic changed: false
- OCR engine changed: false
- External API enabled: false
- Cloud API used: false
- Extraction parser changed: false
- Lab value parser added: false
- Treatment parser added: false
- Safety gate changed: false
- B07 terminology changed: false
- ROUTE-FIX changed: false
- DB schema changed: false
- Command behavior changed: false
- Allowlist changed: false
- Free-form shell added: false

Validation commands:
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

Privacy:
- No raw PHI in report.
- No raw filenames in report.
- No raw document text in report.
- No private absolute paths in report.
- No secrets in report.

Staging safety:
- Private files staged: false
- Source documents staged: false
- Test input files staged: false
- Validation input files staged: false
- Terminology files staged: false
- Runtime DB staged: false
- Unsafe staged files: 0
