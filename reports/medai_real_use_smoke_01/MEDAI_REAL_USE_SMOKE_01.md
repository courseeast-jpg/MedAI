# MEDAI-REAL-USE-SMOKE-01 - Local Operator Smoke Test

## Conclusion

medai_real_use_smoke_01_completed

## Baseline

- baseline_commit: 882fc5391296
- test_type: local_operator_smoke_test
- ui_state: Run & Review polished UI after PARK-17

## Smoke Test Summary

The operator used the Run & Review page after UI polish and upload queue bugfixes. Unsupported formats were selected first and were rejected visually by the uploader. The supported batch used three PDF files, represented here only as safe labels: file_001, file_002, and file_003.

## File Counts

- files_attempted_total: 5
- unsupported_files_observed: 2
- unsupported_file_types: png, docx
- supported_files_processed: 3

## Result Counts

- accepted_count: 0
- needs_review_count: 3
- ocr_scan_review_count: 0
- no_text_found_count: 0
- errors_count: 0

## Safety Results

- external_api_used: false
- cloud_upload_used: false
- privacy_mode: local_only
- auto_acceptance_observed: false
- upload_queue_duplication_observed: false
- unsupported_format_behavior_correct: true
- nothing_auto_accepted: true
- no_crash_observed: true
- uncertain_results_stayed_out_of_accepted: true

## Main Findings

- Supported PDFs processed safely but all were routed to review.
- Lab/test-style PDFs were not confidently classified; document type appeared unknown.
- OCR quality appeared unknown.
- Confidence remained low/moderate, approximately 0.45 to 0.63.
- UI should explain unsupported formats more clearly.
- UI should resolve Needs review versus rejected status inconsistency.

## Safe File Labels

- file_001
- file_002
- file_003
- unsupported_file_type_png
- unsupported_file_type_docx

## Recommended Next Block

MEDAI-LAB-ROUTE-DIAG-01 - Lab PDF Classification and Review Reason Diagnostic

## Possible Follow-up UI Fix

MEDAI-UI-FIX-02 - Unsupported File Type Explanation

## Validation Results

- python scripts/run_medai_ui_ops_panel_validation.py: passed
- python scripts/run_medai_ui_boot_fix_validation.py: passed
- python scripts/run_cka_final_mvp_release_validation.py: passed, 12 of 12 cases, 693 tests reported
- python scripts/run_b07_term01_opt_in_integration_validation.py: passed, 6 of 6 cases
- python scripts/run_medai_route_fix01_validation.py: passed
- python -m pytest tests/test_medai_ui_bugfix_02_readd_after_queue_clear.py: passed, 6 tests
- python -m pytest tests/test_medai_ui_bugfix_01_clear_report_upload_idempotency.py: passed, 6 tests
- python -m pytest tests: not run in this report-only block

## Change Boundary

- clinical_logic_changed: false
- ocr_extractor_changed: false
- safety_gate_changed: false
- b07_terminology_changed: false
- route_fix_changed: false
- db_schema_changed: false
- command_behavior_changed: false
- allowlist_changed: false
- external_api_enabled: false
- private_files_staged: false
- source_documents_staged: false
- test_input_files_staged: false
- real_validation_input_files_staged: false

## Privacy

- no_raw_phi_in_report: true
- no_raw_filenames_in_report: true
- no_private_paths_in_report: true
- no_secrets_in_report: true

