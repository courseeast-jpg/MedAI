# MEDAI-ROUTE-AUDIT-01 - Pending Pipeline/Router Runtime Change Triage

## Scope

This is a focused review package for the pending unstaged runtime changes in:

- `execution/pipeline.py`
- `execution/router.py`

The audit did not edit, stage, revert, reset, restore, check out, delete, or discard either runtime file.

## Baseline

- Branch: `clinical-knowledge-architecture`
- Current HEAD: `6ba33ef`
- HYGIENE-01 reference: `6ba33ef`
- PARK-10 reference: `44db358`
- Audit timestamp: `2026-05-14T09:21:16.8380710-04:00`

## Git Status Summary

The broader worktree remains dirty with 201 `git status --short` entries. The public report does not enumerate medical-looking local filenames or private/local data paths.

Scoped status:

- `execution/pipeline.py`: unstaged modified
- `execution/router.py`: unstaged modified

## Change Cluster Classification

| File | Cluster | Classification | Intended Value | Primary Risk |
| --- | --- | --- | --- | --- |
| `execution/pipeline.py` | PDF text audit state and stage logging | audit/observability only | Make text quality and OCR fallback state visible in audit output | Could expose overly broad metadata if report boundaries are not enforced |
| `execution/pipeline.py` | PII audit state propagation | extraction metadata propagation | Preserve medical-label PII audit facts needed for review/debugging | Could confuse downstream consumers if treated as extraction facts |
| `execution/pipeline.py` | selected extractor and fallback fields | extraction metadata propagation | Preserve route-vs-actual and selected-terminal-result audit trail | Could require report/schema updates before production use |
| `execution/router.py` | selected terminal result metadata | audit/observability only | Expose which extractor was ultimately selected and why | Could create inconsistent audit fields if partially adopted |
| `execution/router.py` | non-empty local result preference over empty fallback | routing behavior change | Prevent empty fallback output from replacing a usable local extraction | Could alter historical route outcomes and accepted/review distribution |
| `execution/router.py` | quota/rate-limit local terminal fallback | quota/rate-limit handling | Retain local result when Gemini quota/rate-limit blocks external route | Could mask external connector failures if audit labels are incomplete |
| `execution/router.py` | terminal empty prevention flags | terminal-empty prevention | Make terminal-empty avoidance explicit and testable | Could be mistaken for an acceptance gate change if not documented |
| `execution/router.py` | fallback chain expansion from spacy to phi3 | fallback behavior change | Allows additional fallback path for low-quality local result | Could broaden routing behavior beyond the original boundary |

## Prior Phase Linkage

The pending changes strongly resemble prior controlled-real-use routing work:

- Preserve `selected_extractor`, `discarded_empty_fallback`, and `fallback_selection_reason`.
- Prefer non-empty local extraction over empty Phi3 fallback in specific terminal-selection cases.
- Preserve the `long_noisy_03` canary behavior.
- Make quota/rate-limit fallbacks visible as audit events.

This is consistent with the earlier controlled real-use finding where readable text could be routed into an empty fallback result. It is not terminology work and is unrelated to B07-TERM behavior.

## Behavior Impact Analysis

If kept:

- Likely benefit: readable/local non-empty output is less likely to be overwritten by an empty fallback.
- Likely benefit: route/fallback audit trails become more explicit.
- Regression risk: route outcomes can change for documents that previously terminated on empty Phi3 or quota-related fallback paths.
- Regression risk: downstream reports may need schema awareness for the new audit fields.

If reverted later:

- Likely benefit: return to PARK-10 runtime baseline for routing and pipeline behavior.
- Regression risk: previously observed empty-fallback replacement behavior may return.
- Regression risk: audit fields useful for controlled-real-use diagnosis would disappear.

## Validation Results

- `python scripts/run_cka_final_mvp_release_validation.py`: passed, 12/12 cases, 693 tests reported, external API used false.
- `python scripts/run_b07_term01_opt_in_integration_validation.py`: passed, 6/6 cases, external API used false.
- `python -m pytest tests/test_phase23_routing_efficiency.py tests/test_connector_orchestration.py tests/test_phase10_hardening.py -q`: passed, 33 tests, 7 warnings.

The focused routing/fallback tests pass with the current pending runtime changes.

## Decision Matrix

| Area | Keep Candidate | Revert Candidate | Needs Test First | Risk Level | Recommended Next Action |
| --- | --- | --- | --- | --- | --- |
| Pipeline PDF/PII audit metadata | Yes | No, unless privacy concern appears | Yes, report-schema/privacy tests | Medium | Keep only in isolated implementation block |
| Pipeline route-selection metadata | Yes | No, unless schema compatibility fails | Yes, downstream audit/report tests | Medium | Keep only with schema and privacy verification |
| Router non-empty local over empty fallback | Yes | Possible if operator wants exact PARK-10 route baseline | Yes, route distribution and canary tests | High | Dedicated implementation block |
| Router quota/rate-limit local fallback | Yes | Possible if quota behavior should remain unchanged | Yes, quota failure and local fallback tests | High | Dedicated implementation block |
| Router spacy-to-phi3 fallback expansion | Unclear | Possible | Yes, targeted route-regression tests | High | Needs operator review before keep |
| Terminal-empty prevention flags | Yes | No, unless naming/schema rejected | Yes, audit-field tests | Medium | Keep with documentation in implementation block |

## Recommended Path

Recommended operator decision: `more_tests_needed_then_keep_candidate_in_isolated_block`.

The current pending changes are likely intentional and already have focused tests passing, but they are runtime behavior changes. They should not be committed as hygiene. The safest next move is a dedicated implementation block that either:

1. Keeps and completes the routing/audit changes with explicit route-distribution, canary, schema, and privacy tests; or
2. Performs a targeted revert after explicit operator approval if the changes are stale.

## Proposed Next Prompt - Keep Path

Start MEDAI-ROUTE-FIX-01 - Controlled Empty-Fallback Prevention Integration.

Use the existing unstaged `execution/pipeline.py` and `execution/router.py` changes as the candidate implementation. Preserve OCR, confidence thresholds, safety gates, B07 terminology behavior, and external API defaults. Add or confirm tests for selected extractor metadata, discarded empty fallback, Gemini quota/rate-limit local fallback, `long_noisy_03` canary preservation, report privacy, and route distribution. Stage and commit only the scoped runtime files, tests, and public reports after validation passes.

## Proposed Next Prompt - Revert Path

Start MEDAI-ROUTE-REVERT-01 - Approved Revert of Pending Pipeline/Router Runtime Changes.

After explicit operator approval, revert only the unstaged changes in `execution/pipeline.py` and `execution/router.py`. Preserve all unrelated dirty files. Run the release validation, B07-TERM validation, and routing smoke tests. Commit only a public revert decision report if no runtime commit is desired.

## Safety And Privacy

- No runtime files were edited by this audit.
- No runtime files were staged by this audit.
- No imports were run.
- No external APIs were used.
- No private terminology artifacts were touched.
- No terminology data, local terminology DB/index, private license acknowledgment, source terminology files, DB/key/private files, PDFs, images, or archives were staged.
- Public report content avoids raw private paths, raw diffs, medical-looking local filenames, PHI, secrets, source terminology rows, and license text.
