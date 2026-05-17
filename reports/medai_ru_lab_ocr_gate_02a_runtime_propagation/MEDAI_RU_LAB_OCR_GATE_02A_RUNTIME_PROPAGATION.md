# MEDAI-RU-LAB-OCR-GATE-02A-FIX

Conclusion: medai_ru_lab_ocr_gate_02a_runtime_propagation_ready

Baseline 02A commit: 5796961

Observed runtime issue: real Run & Review records showed null language_text_visibility and cyrillic_ocr_recommended false.

Root cause: the 02A shadow gate trigger was too strict because it required both medium/high digit density and an explicit table-like flag. The active Run & Review result assembly also only copied existing extractor marker fields, so missing/null marker fields were preserved instead of computing safe marker buckets from available runtime text.

Fix summary: the shadow gate now triggers for substantial native text with either medium/high digit density or a table-like pattern when Cyrillic visibility is zero and OCR was skipped. Run & Review result assembly now defensively computes safe marker metadata when the extractor result lacks marker fields.

## Safety Status

- Marker runtime propagation fixed: true
- Production OCR routing changed: false
- OCR engine changed: false
- OCR fallback executed: false
- Cyrillic OCR recommended marker added: true
- Language text visibility marker added: true
- Review only: true
- Auto-acceptance changed: false
- Confidence thresholds changed: false
- Confidence scoring changed: false
- Extraction parser changed: false
- Lab value parser added: false
- Clinical logic changed: false
- Clinical interpretation added: false
- Medication advice added: false
- DDI logic changed: false
- External API enabled: false
- Cloud API used: false
- Safety gate changed: false
- B07 terminology changed: false
- ROUTE-FIX changed: false
- DB schema changed: false
- Command behavior changed: false
- Allowlist changed: false
- Free-form shell added: false

## Validation Commands

- `python -m pytest tests/test_medai_ru_lab_ocr_gate_02a_runtime_propagation.py -vv`: passed, 8 passed
- `python -m pytest tests/test_medai_ru_lab_ocr_gate_02a_shadow_marker.py -vv`: passed, 11 passed
- `python -m pytest tests/test_medai_ru_lab_ocr_gate_01.py tests/test_medai_ru_lab_text_vis_01.py tests/test_medai_ru_lab_extract_diag_01.py -vv`: passed, 24 passed
- `python -m pytest tests/test_medai_ru_doc_type_02_real_text_path_diagnostic.py tests/test_medai_ru_doc_type_01_russian_detection.py -vv`: passed, 22 passed
- `python -m pytest tests/test_medai_lab_fix_02_general_lab_detection.py tests/test_medai_lab_fix_01_document_type_review_reason.py tests/test_medai_ui_bugfix_02_readd_after_queue_clear.py tests/test_medai_ui_bugfix_01_clear_report_upload_idempotency.py -vv`: passed, 32 passed
- `python scripts/run_medai_ui_ops_panel_validation.py`: passed
- `python scripts/run_medai_ui_boot_fix_validation.py`: passed
- `python scripts/run_cka_final_mvp_release_validation.py`: passed, 12 of 12 cases, 693 tests passed
- `python scripts/run_b07_term01_opt_in_integration_validation.py`: passed, 6 of 6 cases
- `python scripts/run_medai_route_fix01_validation.py`: passed
- `python -m pytest tests`: passed, 2409 passed, 4 skipped, 22 warnings

## Privacy

- Private files staged: false
- Source documents staged: false
- test_input files staged: false
- real_validation_input files staged: false
- No raw PHI: true
- No raw filenames: true
- No raw document text: true
- No private absolute paths: true
- No secrets: true

## Recommendation

Recommended next step: repeat the same small Russian Run & Review smoke test and inspect the raw run record.

Recommended next block if marker fires: MEDAI-RU-LAB-OCR-GATE-02B - Local Cyrillic OCR Fallback.

Fallback: marker_absent_no_02b.
