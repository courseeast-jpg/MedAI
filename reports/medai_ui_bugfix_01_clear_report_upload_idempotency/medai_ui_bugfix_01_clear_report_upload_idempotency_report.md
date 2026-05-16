# MEDAI-UI-BUGFIX-01 Report

Conclusion: medai_ui_bugfix_01_clear_report_upload_idempotency_ready

The bug was caused by Streamlit preserving selected uploader files across reruns. Clear last report triggered a rerun, and the upload persistence path could save those selected files again.

The fix makes upload persistence idempotent per uploader session using safe file fingerprints. Clear last report now remains isolated to clearing the latest visible report/current run state and does not reset upload fingerprints or touch the queue. Remove queued files still clears the queue and now resets the uploader tracking/key so the operator can start fresh.

Required focused UI tests and required safety validations passed. Full pytest was not rerun in this block; PARK-16 full suite passed immediately before this bugfix series.

Privacy status: clean. This report includes no raw PHI, no raw filenames, no private absolute paths, and no secrets.
