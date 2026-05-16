# MEDAI-UI-BUGFIX-02 - Re-add After Queue Clear

## Conclusion

medai_ui_bugfix_02_readd_after_queue_clear_ready

## Scope

This block fixed Run & Review upload tracking after the operator removes queued files. It does not change document processing, extraction, routing, review package logic, safety gates, B07 terminology behavior, ROUTE-FIX behavior, command behavior, command allowlists, database schema, or external API settings.

## Observed Bug

After the queue was cleared, selecting the same small batch again could leave the uploader display and MedAI queue out of sync: the uploader showed selected files while Documents waiting / Files ready remained zero.

## Root Cause Summary

The duplicate-save prevention from the prior bugfix used per-session upload fingerprints. That correctly prevented Clear last report from re-saving selected files, but the tracking needed an explicit upload-widget generation boundary so a queue clear starts a clean upload session.

## Fix Summary

Upload persistence is now scoped to the current uploader generation. Remove queued files clears the queue, clears persisted upload fingerprints, and advances the uploader generation. Clear last report does not alter the queue, upload fingerprints, or uploader generation.

## Safety Assertions

- clear_last_report_changes_queue: false
- repeated_clear_last_report_duplicates_files: false
- remove_queued_files_resets_upload_tracking: true
- same_files_can_be_readded_after_queue_clear: true
- uploader_queue_state_consistent: true
- document_processing_changed: false
- review_package_logic_changed: false
- review_bucket_logic_changed: false
- clinical_logic_changed: false
- ocr_extractor_changed: false
- safety_gate_changed: false
- cka_safety_behavior_changed: false
- b07_terminology_changed: false
- route_fix_changed: false
- command_behavior_changed: false
- allowlist_changed: false
- free_form_shell_added: false
- external_api_enabled: false
- db_schema_changed: false
- private_files_staged: false
- source_documents_staged: false
- terminology_files_staged: false
- runtime_db_staged: false
- unsafe_staged_files: 0

## Validation Results

- python -m pytest tests/test_medai_ui_bugfix_02_readd_after_queue_clear.py: passed, 6 tests
- python -m pytest tests/test_medai_ui_bugfix_01_clear_report_upload_idempotency.py: passed, 6 tests
- python -m pytest tests/test_medai_ui_polish_07_header_audit_advanced_actions.py: passed, 5 tests
- python -m pytest tests/test_medai_ui_polish_06_streamlit_chrome_sidebar.py: passed, 5 tests
- python -m pytest tests/test_medai_ui_polish_05_run_review_consolidation.py: passed, 5 tests
- python -m pytest tests/test_medai_ui_ops_panel.py: passed, 13 tests
- python scripts/run_medai_ui_ops_panel_validation.py: passed
- python scripts/run_medai_ui_boot_fix_validation.py: passed
- python scripts/run_cka_final_mvp_release_validation.py: passed, 12 of 12 cases, 693 tests reported
- python scripts/run_b07_term01_opt_in_integration_validation.py: passed, 6 of 6 cases
- python scripts/run_medai_route_fix01_validation.py: passed
- python -m pytest tests: passed, 2324 passed, 4 skipped, 22 warnings

## Privacy

This report contains no raw PHI, no raw filenames, no private absolute paths, no secrets, no source terminology rows, and no database contents.
