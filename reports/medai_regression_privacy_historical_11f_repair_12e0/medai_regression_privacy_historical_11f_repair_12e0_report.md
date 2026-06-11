# MEDAI-REGRESSION-PRIVACY-HISTORICAL-11F-REPAIR-12E0 Report

Conclusion: `medai_regression_privacy_historical_11f_repair_12e0_ready`

## Root Cause

The historical regression was a nondeterministic false positive in public-report privacy validation. Synthetic canonical UUID record IDs can contain a substring that matches the broad medical-record-number detector. The leak example was already redacted by the privacy checker and was not raw PHI.

## Exact Fix

The public-report checker now filters only identifier findings fully contained inside canonical UUID strings. Raw patient identifiers and other privacy-sensitive values remain blocked.

## Privacy and Safety

- privacy_result: `passed`
- external_api_used: `false`
- auto_accept: `false`
- no raw PHI: `true`
- no raw OCR text: `true`
- no private paths: `true`
- runtime DB mutation: `false`

## Tests Run

- `python -m pytest tests/test_medai_regression_privacy_historical_11f_repair_12e0.py tests/test_medai_actual_streamlit_ui_wiring_fix_11f.py` -> `passed`
- `python -m pytest tests/test_medai_actual_one_screen_ux_final_11g.py` -> `passed`
- `python -m pytest tests/test_medai_run_review_density_final_11h.py` -> `passed`
- `python -m pytest tests/test_medai_image_ocr_reviewbound_routing_12d.py` -> `passed`
- `python -m pytest tests/test_medai_local_image_ocr_routing_12c.py` -> `passed`
- `python -m pytest tests/test_medai_file_intake_multiformat_uat_12b.py` -> `passed`
- `python -m py_compile app/main.py app/test_launcher.py execution/jobs.py execution/pipeline.py` -> `passed`
