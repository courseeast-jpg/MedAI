# MEDAI-RU-DOC-TYPE-01 Russian Document Type Detection

Conclusion: medai_ru_doc_type_01_russian_detection_ready

## Scope

This block adds Russian/Cyrillic metadata-only document type detection for operator-facing labels and review reasons.

Supported document types:
- Lab result
- Urinalysis
- Treatment plan
- Medication plan
- Unknown

## Behavior Changed

- Russian lab-style cues can label a document as Lab result.
- Russian urinalysis-style cues can label a document as Urinalysis.
- Russian treatment-plan cues can label a document as Treatment plan.
- Russian medication-plan cues can label a document as Medication plan.
- Low-confidence lab and urinalysis documents remain review-bound with clearer review reasons.
- Treatment-plan and medication-plan documents remain review-bound with explicit human-review guidance.

## Behavior Not Changed

- Auto-acceptance unchanged.
- Confidence thresholds unchanged.
- Confidence scoring unchanged.
- Clinical logic unchanged.
- Clinical interpretation not added.
- Medication advice not added.
- DDI logic unchanged.
- OCR engine unchanged.
- External APIs not enabled.
- Extraction parser unchanged.
- Lab value parser not added.
- Treatment parser not added.
- Safety gates unchanged.
- B07 terminology behavior unchanged.
- ROUTE-FIX behavior unchanged.
- DB schema unchanged.
- Operator command behavior unchanged.
- Command allowlist unchanged.
- Free-form shell input not added.

## Validation Summary

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
- `python -m pytest tests`: passed, 2356 passed, 4 skipped, 22 warnings.

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
