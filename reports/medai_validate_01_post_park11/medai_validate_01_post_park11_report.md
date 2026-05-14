# MEDAI-VALIDATE-01 Report

Conclusion: `full_suite_validation_failed_unrelated_existing_issues`

Branch: `clinical-knowledge-architecture`

HEAD: `2c3126a`

## Full Pytest Result

`python -m pytest tests` failed:

- 2205 passed
- 5 failed
- 4 skipped
- 22 warnings
- 2214 collected

The remaining requested validations were not run because the full suite failed first. No runtime code was patched.

## Failure Classification

Two failures are stale B07 planning expectations after the approved B07-TERM-01 implementation:

- `tests/test_b07_term_opt_in_planning.py::test_no_runtime_implementation_file_created`
- `tests/test_b07_term_opt_in_planning.py::test_validation_script_runs`

Three failures are manual-license state sensitive after local RxNorm and LOINC acknowledgment:

- `tests/test_cka_term01_real_terminology_readiness.py::TestLicenseGate::test_real_systems_blocked_without_ack[rxnorm]`
- `tests/test_cka_term01_real_terminology_readiness.py::TestLicenseGate::test_real_systems_blocked_without_ack[loinc]`
- `tests/test_cka_term01a_intake_automation.py::TestReadinessChecker::test_template_does_not_count_as_ack`

These failures are not related to ROUTE-FIX-01 routing/fallback behavior.

## Safety

- Validation-only block.
- No runtime files edited.
- No routing/fallback logic changed.
- No B07 terminology behavior changed.
- No clinical logic changed.
- No OCR/extractor/safety gates changed.
- No imports run.
- No external APIs used.
- No private terminology files, runtime DB/index files, source terminology files, DB/key/private/PDF/image/archive files staged.

## Recommended Next Action

Open a dedicated QA isolation block to update stale B07 planning tests and isolate manual-license state in terminology readiness tests. Keep ROUTE-FIX-01 unchanged for these failures.
