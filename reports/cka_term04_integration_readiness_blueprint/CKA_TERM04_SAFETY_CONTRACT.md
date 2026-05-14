# CKA-TERM-04 Safety Contract

## Scope

This contract governs future terminology-backed integration work after TERM-02 import and TERM-03 QA. TERM-04 itself is design-only and does not modify runtime clinical behavior.

## Always-On Boundaries

- Local-only operation remains required.
- External APIs remain disabled for terminology work.
- Source terminology files remain private and uncommitted.
- Private license acknowledgment remains private and uncommitted.
- Local terminology DB/index files remain private and uncommitted.
- Public reports may include aggregate counts, safe status labels, and safe hashes only.

## Lookup Behavior Requirements

- Exact matches may be reported as exact lookup results.
- Source filters must prevent cross-system contamination.
- Unknown terms must return unmapped.
- Ambiguous terms must return ambiguous/manual-review.
- Repeated lookup must be deterministic.
- Query normalization must be deterministic.
- No code may be invented.

## Clinical Behavior Prohibitions

Terminology lookup must not:
- must not generate clinical advice.
- Generate prescribing or dosing advice.
- Promote a hypothesis.
- must not clear or downgrade DDI status.
- Override existing confidence gates.
- Override existing review gates.
- Convert a review result to accepted.
- Change OCR or extraction routing.

## B07 Contract

B07 integration remains disabled until an explicit future opt-in block. Any future B07 terminology work must preserve:
- Hypothesis-only status by default.
- Manual review for ambiguous terms.
- Unmapped status for unknown terms.
- No DDI clearing.
- No automatic accepted clinical fact creation.

## Report Privacy Contract

Public reports must not contain:
- Raw private paths.
- License text.
- Source terminology rows.
- Private file contents.
- PHI.
- Secrets.
- Local DB file paths.

## Required Stop Conditions

Stop immediately if:
- A private terminology file is staged.
- A local DB/index file is staged.
- A public report leaks private path, license text, source row content, PHI, or secret-like material.
- An external API call is attempted.
- Unknown input receives an invented code.
- Ambiguous input is silently resolved.
- B07 behavior changes without explicit approval.
