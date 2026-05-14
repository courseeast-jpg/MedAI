# MEDAI-ROUTE-AUDIT-01 Report

Conclusion: `pending_route_runtime_changes_need_operator_decision`

Branch: `clinical-knowledge-architecture`

HEAD: `6ba33ef`

HYGIENE-01 reference: `6ba33ef`

## Summary

The pending changes in `execution/pipeline.py` and `execution/router.py` are likely intentional runtime work, not generated noise. They are coherent enough to pass focused routing/fallback tests, but they alter routing and fallback behavior and should be handled in a dedicated implementation block.

## Classifications

| Change Area | Classification | Risk | Recommendation |
| --- | --- | --- | --- |
| Pipeline PDF text audit metadata | audit/observability only | Medium | Keep candidate with privacy/schema tests |
| Pipeline PII audit propagation | extraction metadata propagation | Medium | Keep candidate with report-boundary tests |
| Pipeline selected extractor fields | extraction metadata propagation | Medium | Keep candidate with downstream schema checks |
| Router selected terminal metadata | audit/observability only | Medium | Keep candidate with audit-field tests |
| Router non-empty local over empty fallback | routing behavior change | High | Dedicated implementation block |
| Router quota/rate-limit local fallback | quota/rate-limit handling | High | Dedicated implementation block |
| Router terminal-empty prevention | terminal-empty prevention | Medium | Keep candidate with canary tests |
| Router spacy-to-phi3 fallback expansion | fallback behavior change | High | Operator review before keep |

## Intended Value

The changes appear designed to prevent a usable local extraction from being replaced by an empty fallback result, especially around Gemini fallback and Phi3 terminal paths. They also make route selection, fallback reasons, text quality, OCR fallback state, and PII audit state more visible for operator review.

## Regression Risks

Keeping the changes could shift route outcomes for documents that previously terminated on an empty fallback. It could also require downstream report or schema updates for new audit fields.

Reverting the changes could restore the prior empty-fallback replacement risk and remove useful audit fields for controlled-real-use diagnosis.

## Validation Results

- `python scripts/run_cka_final_mvp_release_validation.py`: passed, 12/12 cases, 693 tests reported.
- `python scripts/run_b07_term01_opt_in_integration_validation.py`: passed, 6/6 cases.
- `python -m pytest tests/test_phase23_routing_efficiency.py tests/test_connector_orchestration.py tests/test_phase10_hardening.py -q`: passed, 33 tests, 7 warnings.

## Recommended Operator Decision

Recommended path: `more_tests_needed_then_keep_candidate_in_isolated_block`.

The route changes should not be committed as incidental hygiene. The safest next action is a dedicated controlled routing implementation block that keeps the candidate changes only after confirming:

- selected extractor metadata
- discarded empty fallback behavior
- Gemini quota/rate-limit local fallback behavior
- `long_noisy_03` canary preservation
- route distribution stability
- report/schema compatibility
- public report privacy cleanliness

## Proposed Keep Prompt

Start MEDAI-ROUTE-FIX-01 - Controlled Empty-Fallback Prevention Integration.

Use the existing unstaged `execution/pipeline.py` and `execution/router.py` changes as the candidate implementation. Preserve OCR, confidence thresholds, safety gates, B07 terminology behavior, and external API defaults. Add or confirm tests for selected extractor metadata, discarded empty fallback, Gemini quota/rate-limit local fallback, `long_noisy_03` canary preservation, report privacy, and route distribution. Stage and commit only scoped runtime files, tests, and public reports after validation passes.

## Proposed Revert Prompt

Start MEDAI-ROUTE-REVERT-01 - Approved Revert of Pending Pipeline/Router Runtime Changes.

After explicit operator approval, revert only the unstaged changes in `execution/pipeline.py` and `execution/router.py`. Preserve all unrelated dirty files. Run release validation, B07-TERM validation, and routing smoke tests. Commit only a public revert decision report if no runtime commit is desired.

## Safety

- Runtime files were not edited by this audit.
- Runtime files were not staged by this audit.
- No imports were run.
- No external APIs were used.
- No private terminology artifacts were touched.
- No terminology data, local terminology DB/index, private license acknowledgment, source terminology files, DB/key/private files, PDFs, images, or archives were staged.
- Public report content excludes raw private paths, raw diffs, medical-looking local filenames, PHI, secrets, source terminology rows, and license text.
