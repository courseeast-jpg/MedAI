# MEDAI-UI-BUGFIX-01 Clear Report Upload Idempotency

Conclusion: medai_ui_bugfix_01_clear_report_upload_idempotency_ready

## Observed Bug

Clear last report caused Files ready to increase because the Streamlit uploader selection remained populated across reruns.

## Root Cause Summary

The upload persistence path ran whenever the file uploader returned selected files. Streamlit preserves file uploader state across reruns, so unrelated actions such as Clear last report could re-enter the upload persistence path and write the same selected files again.

## Fix Summary

- Added per-session uploaded file fingerprints based on safe filename, size, and content hash.
- Persisted uploads only when a fingerprint has not already been saved in the current uploader session.
- Kept Clear last report isolated to latest report removal and current run state clearing.
- Kept Remove queued files behavior while resetting upload persistence tracking and the uploader widget key.
- Preserved duplicate filename handling for genuinely distinct uploads.

## Safety Boundary

- Clear last report changes queue: false
- Upload persistence idempotent: true
- Remove queued files behavior changed: false
- Document processing changed: false
- Review package logic changed: false
- Review bucket logic changed: false
- Clinical logic changed: false
- OCR/extractor changed: false
- Safety gate changed: false
- CKA safety behavior changed: false
- B07 terminology changed: false
- Route-fix changed: false
- Command behavior changed: false
- Allowlist changed: false
- Free-form shell added: false
- External API enabled: false
- DB schema changed: false
- Private files staged: false
- Source documents staged: false
- Terminology files staged: false
- Runtime DB staged: false
- Unsafe staged files: 0

## Validation Results

- python -m pytest tests/test_medai_ui_bugfix_01_clear_report_upload_idempotency.py: passed, 6 tests
- python -m pytest tests/test_medai_ui_polish_07_header_audit_advanced_actions.py tests/test_medai_ui_polish_06_streamlit_chrome_sidebar.py tests/test_medai_ui_polish_05_run_review_consolidation.py tests/test_medai_ui_polish_04_navigation_advanced_mode.py -q: passed, 22 tests
- python -m pytest tests/test_medai_ui_polish_03_operator_control_panel.py tests/test_medai_ui_polish_02_review_package.py tests/test_medai_ui_polish_01_current_run.py tests/test_medai_ui_ops_panel.py -q: passed, 34 tests
- python scripts/run_medai_ui_ops_panel_validation.py: passed
- python scripts/run_medai_ui_boot_fix_validation.py: passed
- python scripts/run_cka_final_mvp_release_validation.py: passed, 12/12 cases, 693 tests reported
- python scripts/run_b07_term01_opt_in_integration_validation.py: passed, 6/6 cases
- python scripts/run_medai_route_fix01_validation.py: passed
- python -m pytest tests: not run in this block; PARK-16 full suite passed immediately before this bugfix series

## Privacy

No raw PHI, no raw filenames, no private absolute paths, and no secrets are included.

## Recommended Smoke Test

In Run & Review, select documents once, note Files ready, click Clear last report, and confirm Files ready does not increase.
