# MEDAI-HYGIENE-01 - Post PARK-10 Worktree Audit

## Scope

This audit inspected the unstaged worktree after PARK-10 without changing runtime behavior. The only code diffs inspected were:

- `execution/pipeline.py`
- `execution/router.py`

No files were staged, committed, reset, restored, checked out, deleted, or discarded during the audit step before report creation.

## Baseline

- Branch: `clinical-knowledge-architecture`
- HEAD: `44db358`
- PARK-10 reference: `44db358`
- Audit timestamp: `2026-05-14T09:14:13.9430421-04:00`

## Worktree Status Summary

`git status --short` showed a dirty worktree with 201 entries. The dirty set includes tracked generated reports/artifacts, several tracked source/test files outside the requested audit scope, untracked local folders, and local validation/test artifacts. The public report intentionally does not list medical-looking local filenames or private/local data paths.

The two requested runtime files are both unstaged:

- `execution/pipeline.py`
- `execution/router.py`

## Safe Diff Summary

### `execution/pipeline.py`

Diff size: 45 insertions, 2 deletions.

Observed themes:

- Adds transient PDF text audit state to the pipeline object.
- Adds transient PII audit state to the pipeline object.
- Emits additional stage-log metadata for PDF text extraction.
- Adds routing-selection audit fields to extracted output.
- Switches PDF text extraction from a text-only helper to an audit-returning helper.
- Copies selected PII and text-quality audit fields into extracted metadata.

Classification: `likely intentional pending work`.

Reasoning: The changes are substantive behavior/audit changes, not generated noise or formatting-only churn. They appear related to prior routing and controlled-real-use audit work, but they were not part of PARK-10 and remain unstaged.

Recommended operator action: `investigate before keep_or_revert_decision`.

### `execution/router.py`

Diff size: 287 insertions, 22 deletions.

Observed themes:

- Extends routed extraction metadata with selected-extractor and fallback-selection fields.
- Adds terminal-empty-prevention and primary/fallback extractor metadata.
- Adds special handling for quota/rate-limit fallback behavior.
- Adds logic to prefer a non-empty local result over an empty fallback result in specific terminal-selection paths.
- Adds helper methods for quota-safe local terminal selection and best non-empty local result selection.
- Changes some fallback-routing and degradation handling branches.

Classification: `likely intentional pending work`.

Reasoning: The changes are large and runtime-significant. They are not generated artifacts. They affect routing/fallback behavior and therefore require explicit operator approval or a dedicated implementation block before they should be kept or reverted.

Recommended operator action: `investigate before keep_or_revert_decision`.

## Validation Results

- `python scripts/run_cka_final_mvp_release_validation.py`: passed, 12/12 validation cases passed, 693 tests reported, external API used false.
- `python scripts/run_b07_term01_opt_in_integration_validation.py`: passed, 6/6 validation cases passed, external API used false.

## Safety And Privacy

- Runtime behavior was not edited by this audit.
- No imports were run.
- No external APIs were used.
- No private terminology artifacts were touched.
- No terminology data, local terminology DB/index, private license acknowledgment, source terminology files, DB/key/private files, PDFs, images, or archives were staged.
- Public report content avoids raw private paths, raw diffs, medical-looking local filenames, PHI, secrets, source terminology rows, and license text.

## Operator Decision

Recommended decision: keep both `execution/pipeline.py` and `execution/router.py` unstaged until the operator chooses one of these paths:

1. Create a dedicated implementation/review block for the pending routing and audit changes.
2. Approve a targeted revert later if these changes are confirmed stale.
3. Continue investigation with focused tests before deciding.

No revert or cleanup should happen automatically from this hygiene report.
