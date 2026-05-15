# MEDAI-UI-POLISH-04 Navigation Advanced Mode Cleanup

## Conclusion

`medai_ui_polish_04_navigation_advanced_mode_ready`

Navigation was simplified so the daily operator pages appear by default and advanced/admin/audit pages are shown only when the operator enables `Show advanced tools`.

## Case J Diagnosis

- CKA final MVP case J initially failed because two CKA-B09 tests were driven by a stale validation-script expectation.
- Exact failing tests identified:
  - `tests/test_cka_block09_operator_ui.py::TestValidationScript::test_validation_script_runs_cleanly`
  - `tests/test_cka_block09_operator_ui.py::TestValidationScript::test_validation_all_14_cases_pass`
- Classification: `stale_ui_label_expectations`
- Fix: update the CKA-B09 validation case to verify the same safety dashboard integration under the new `Safety & Governance` advanced-mode navigation.
- Final MVP validation after fix: `passed_12_of_12`

## Scope

- Changed scope: navigation/sidebar UI wording and advanced-mode visibility only
- Backend behavior changed: false
- Page behavior changed: false
- Command behavior changed: false
- Allowlist changed: false
- Free-form shell added: false
- Clinical logic changed: false
- OCR/extractor changed: false
- Safety gate changed: false
- Review bucket logic changed: false
- B07 terminology changed: false
- ROUTE-FIX changed: false
- External API enabled: false
- DB schema changed: false

## Navigation Changes

Default navigation:
- Current Run
- Review Package
- Operator Control Panel

Advanced navigation after `Show advanced tools`:
- Validation Batch Audit
- Validation History
- Safety & Governance
- Terminology Admin

Sidebar wording:
- Knowledge base
- Active
- Draft facts
- Medical connector active
- Enrichment enabled

## Validation Results

| Command | Result |
| --- | --- |
| CKA B01-B10 Case J pytest subset with `-vv --tb=short` | passed, 693 tests |
| `python -m pytest tests/test_medai_ui_polish_04_navigation_advanced_mode.py -vv` | passed, 7 tests |
| `python -m pytest tests/test_medai_ui_polish_03_operator_control_panel.py -vv` | passed, 7 tests |
| `python -m pytest tests/test_medai_ui_ops_panel.py -vv` | passed, 13 tests |
| `python -m pytest tests/test_medai_ui_polish_02_review_package.py -vv` | passed, 6 tests |
| `python -m pytest tests/test_medai_ui_polish_01_current_run.py -vv` | passed, 8 tests |
| `python -m pytest tests/test_phase52_operator_ui_redesign.py -vv` | passed, 10 tests |
| `python -m pytest tests/test_phase49_operator_ui.py -vv` | passed, 6 tests |
| `python -m pytest tests/test_cka_term01e_operator_readiness_ui.py -vv` | passed, 9 tests |
| `python scripts/run_medai_ui_ops_panel_validation.py` | passed |
| `python scripts/run_medai_ui_boot_fix_validation.py` | passed |
| `python scripts/run_cka_final_mvp_release_validation.py` | passed, 12/12 cases |
| `python scripts/run_b07_term01_opt_in_integration_validation.py` | passed, 6/6 cases |
| `python scripts/run_medai_route_fix01_validation.py` | passed |
| `python -m pytest tests` | passed, 2297 passed, 4 skipped, 22 warnings |

## Safety And Privacy

- Private files staged: false
- Terminology files staged: false
- Runtime DB staged: false
- Unsafe staged files: 0
- Raw PHI included: false
- Raw filenames included: false
- Private absolute paths included: false
- Secrets included: false

## Next Recommended Action

Park the UI polish state or continue only with a new approval-gated UI cleanup block.
