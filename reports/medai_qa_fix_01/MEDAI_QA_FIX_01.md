# MEDAI-QA-FIX-01 - Full-Suite Drift Isolation

## Summary

- Block: `MEDAI-QA-FIX-01`
- Conclusion: `full_suite_drift_isolated`
- Branch: `clinical-knowledge-architecture`
- Starting HEAD: `a077b1c`
- Report timestamp: `2026-05-14T11:04:24.0842745-04:00`

This block fixed stale and state-sensitive tests only. Runtime behavior was not changed.

## Root Cause

The previous full-suite failure had two independent causes:

1. B07 planning tests still asserted the pre-implementation condition that the opt-in B07 terminology shim must not exist. That was correct during B07-TERM-PLAN-01, but stale after explicitly approved B07-TERM-01.
2. Terminology readiness negative tests consulted the real local operator acknowledgment state. That made no-ack tests fail after local RxNorm and LOINC acknowledgment became present.

No failure was related to ROUTE-FIX-01 routing/fallback behavior.

## Files Changed

- `tests/test_b07_term_opt_in_planning.py`
- `scripts/run_b07_term_opt_in_planning_validation.py`
- `tests/test_cka_term01_real_terminology_readiness.py`
- `tests/test_cka_term01a_intake_automation.py`

## Fixes

- Updated B07 planning tests to preserve the design-only planning contract while recognizing the later approved B07-TERM-01 implementation.
- Updated the B07 planning validation script to treat `b07_term_opt_in.py` as approved only when the B07-TERM-01 public report proves default-off, hypothesis-only, read-only safety.
- Isolated real-system no-ack tests with explicit temporary missing acknowledgment paths.
- Isolated the template-readiness negative test from real local operator acknowledgment state.

## Validation Results

- `python -m pytest tests/test_b07_term_opt_in_planning.py -vv`: passed, 8 tests.
- `python -m pytest tests/test_cka_term01_real_terminology_readiness.py tests/test_cka_term01a_intake_automation.py -vv`: passed, 102 tests.
- `python -m pytest tests`: passed, 2210 tests, 4 skipped, 22 warnings.
- `python scripts/run_b07_term01_opt_in_integration_validation.py`: passed, 6/6 cases.
- `python scripts/run_medai_route_fix01_validation.py`: passed with `medai_route_fix01_ready`.
- `python scripts/run_cka_final_mvp_release_validation.py`: passed, 12/12 cases, 693 tests reported.

## Safety

- Runtime behavior changed: false.
- ROUTE-FIX-01 routing/fallback logic changed: false.
- B07-TERM-01 runtime behavior changed: false.
- Clinical logic changed: false.
- OCR/extractor/safety gates changed: false.
- New imports run: false.
- External APIs used: false.
- Private/manual-license state isolated in negative tests: true.
- No private terminology files, runtime DB/index files, source terminology files, DB/key/private/PDF/image/archive files were staged.

## Recommended Next Action

Create a parking snapshot for the clean full-suite state if this QA fix is accepted.
