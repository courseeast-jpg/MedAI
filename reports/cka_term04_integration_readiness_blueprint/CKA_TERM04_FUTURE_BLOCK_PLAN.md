# CKA-TERM-04 Future Block Plan

## A. TERM-05 Synthetic Read-Only Terminology Adapter

Goal: Build a read-only adapter using synthetic terminology fixtures only.

Allowed:
- Synthetic store only.
- Read-only lookup.
- Safe lookup result schema.
- Feature flags default off.

Blocked:
- Private-store access.
- B07 integration.
- Clinical writes.

Exit criteria:
- Exact, unknown, ambiguous, deterministic, and source-filter tests pass.
- Public reports privacy-clean.

## B. TERM-06 Private-Store Read-Only Adapter Validation

Goal: Validate the TERM-05 adapter against the private TERM-02 local store without changing runtime behavior.

Allowed:
- Read-only private-store lookup.
- Aggregate metrics.
- Safe report fields.

Blocked:
- Runtime clinical use.
- Automatic annotations.
- B07 integration.
- Public source row output.

Exit criteria:
- TERM-03 QA still passes.
- Adapter read-only validation passes.
- No private files or DB/index files staged.

## C. TERM-07 UI-Only Terminology Lookup Panel

Goal: Provide an operator-only panel for local terminology lookup visibility.

Allowed:
- Local-only UI lookup.
- Safe display of status, source system, code count, and ambiguity status.
- No raw source file content.

Blocked:
- Runtime clinical writes.
- Automatic extraction changes.
- B07 integration.

Exit criteria:
- UI cannot crash if private store missing.
- Local-only/default-off behavior preserved.
- Public reports privacy-clean.

## D. TERM-08 Hypothesis-Only Coding Annotation Pilot

Goal: Pilot terminology-backed annotations as explicitly marked hypotheses.

Allowed:
- Opt-in hypothesis annotations.
- Manual review requirement.
- Clear reason codes.

Blocked:
- Accepted clinical fact promotion.
- DDI status clearing.
- Dosing or prescribing advice.
- Silent ambiguity resolution.

Exit criteria:
- Feature flag off preserves current behavior.
- Hypothesis-only tests pass.
- Unknown and ambiguous safety tests pass.

## E. B07-TERM Opt-In Integration

Goal: Consider B07 terminology integration only after explicit operator approval and successful TERM-05 through TERM-08 evidence.

Allowed:
- Explicitly approved opt-in only.
- Hypothesis-preserving integration.
- Full rollback flag.

Blocked:
- Default-on behavior.
- Automatic acceptance.
- must not generate clinical advice.
- must not clear or downgrade DDI status.

Exit criteria:
- New approval prompt exists.
- Full regression suite passes.
- Safety red-team passes.
- Public reports remain privacy-clean.
