# MEDAI-REGRESSION-PRIVACY-HISTORICAL-11F-REPAIR-12E0

## Result

The nested 11F regression chain is repaired.

## Root Cause

The public-report privacy checker scanned synthetic run payloads containing canonical UUID record IDs. Some UUIDs can contain a short letter-plus-number substring that matches the broad historical medical-record-number detector. The reported leak example was a redacted privacy-check token, not raw PHI, but it caused the 11B privacy assertion to fail nondeterministically.

## Exact Fix

The public-report checker now ignores identifier findings only when the finding is fully contained inside a canonical UUID string. Raw patient identifiers, names, dates of birth, facility names, private paths, secrets, raw OCR text, and non-UUID identifier values remain blocked.

## Safety

- privacy_result: `passed`
- external_api_used: `false`
- auto_accept: `false`
- no raw PHI: `true`
- no raw OCR text: `true`
- no private paths: `true`
- runtime DB mutation: `false`

## Tests Run

- `python -m pytest tests/test_medai_regression_privacy_historical_11f_repair_12e0.py tests/test_medai_actual_streamlit_ui_wiring_fix_11f.py`
- `python -m pytest tests/test_medai_actual_one_screen_ux_final_11g.py`
- `python -m pytest tests/test_medai_run_review_density_final_11h.py`
- `python -m pytest tests/test_medai_image_ocr_reviewbound_routing_12d.py`
- `python -m pytest tests/test_medai_local_image_ocr_routing_12c.py`
- `python -m pytest tests/test_medai_file_intake_multiformat_uat_12b.py`
- `python -m py_compile app/main.py app/test_launcher.py execution/jobs.py execution/pipeline.py`
