# MEDAI-DOC-TYPE-EVAL-01

- Conclusion: `medai_doc_type_eval_01_ready`
- Total files evaluated: `0`
- External API used count: `0`
- Auto-accept allowed count: `0`
- OCR fallback used count: `0`
- Ambiguous family conflict count: `0`

## Count By Predicted Document Type

- Lab result: `0`
- Imaging report: `0`
- Treatment plan: `0`
- Medication plan: `0`
- Clinical note: `0`
- Discharge summary: `0`
- Unknown: `0`
- Referral / Order: `0`
- Procedure report: `0`
- Pathology report: `0`
- Administrative / Insurance: `0`

## Count By Review Status


## Recommended Next Actions

- cue-pack update needed: `0`
- conflict-resolution update needed: `0`
- UI-only issue: `0`
- leave Unknown/manual review: `0`

## Anonymous Per-File Table

- No supported files evaluated.

## Report Generation Note

This committed report was generated without committing or reading private corpus documents. Operators can run the same script locally with `--input-dir` against a private corpus to produce anonymized `file_id` summaries.

## Validation Results

- `python -m pytest tests/test_medai_doc_type_eval_01_batch_harness.py -q` — passed, 5 passed, 1 warning.
- `python -m pytest tests/test_medai_doc_type_eval_01_batch_harness.py tests/test_medai_doc_type_family_01_fix2_imaging_treatment_conflict.py tests/test_medai_doc_type_family_01_fix_runtime_imaging_propagation.py tests/test_medai_doc_type_family_01.py tests/test_medai_ui_run_review_polish_01_fix_state_consistency.py tests/test_medai_ui_run_review_polish_01.py -q` — passed, 39 passed, 1 warning.
- `python -m pytest tests/test_medai_ru_treatment_doc_type_01_fix2_always_emit_diagnostic.py tests/test_medai_ru_treatment_doc_type_01_fix_runtime_propagation.py tests/test_medai_ru_treatment_doc_type_01.py tests/test_medai_ru_lab_ocr_gate_02b_fix2_real_format_cue_coverage.py tests/test_medai_ru_lab_ocr_gate_02b_fix_fallback_cue_diagnostic.py tests/test_medai_ru_lab_ocr_gate_02b_local_cyrillic_fallback.py tests/test_medai_ru_lab_ocr_gate_02a_runtime_propagation.py tests/test_medai_ru_lab_ocr_gate_02a_shadow_marker.py tests/test_medai_ru_lab_ocr_gate_01.py tests/test_medai_ru_lab_text_vis_01.py tests/test_medai_ru_lab_extract_diag_01.py tests/test_medai_ru_doc_type_02_real_text_path_diagnostic.py tests/test_medai_ru_doc_type_01_russian_detection.py tests/test_medai_lab_fix_02_general_lab_detection.py tests/test_medai_lab_fix_01_document_type_review_reason.py tests/test_medai_ui_bugfix_02_readd_after_queue_clear.py tests/test_medai_ui_bugfix_01_clear_report_upload_idempotency.py -q` — passed, 140 passed.
- `python scripts/run_medai_ui_ops_panel_validation.py` — passed.
- `python scripts/run_medai_ui_boot_fix_validation.py` — passed.
- `python scripts/run_cka_final_mvp_release_validation.py` — passed, 12/12 cases, 693 tests passed.
- `python scripts/run_b07_term01_opt_in_integration_validation.py` — passed, 6/6 cases.
- `python scripts/run_medai_route_fix01_validation.py` — passed.
- `python -m pytest tests` — passed, 2491 passed, 4 skipped, 22 warnings in 1662.08 seconds.

## Safety / Privacy

- No raw OCR text.
- No raw document text.
- No raw filenames.
- No private paths.
- No PHI.
- No secrets.
- Source documents staged: false.
- External APIs enabled: false.
