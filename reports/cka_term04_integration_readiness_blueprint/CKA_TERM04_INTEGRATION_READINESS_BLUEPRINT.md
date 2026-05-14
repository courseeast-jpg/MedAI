# CKA-TERM-04 Integration Readiness Blueprint

## Current Capability Boundary

CKA-TERM-02 completed a controlled local import for acknowledged local rxnorm and loinc files. CKA-TERM-03 completed read-only QA over the resulting private local terminology store. This block does not activate terminology-backed clinical behavior.

Current capabilities:
- Local rxnorm and loinc terminology records exist in a private, gitignored local store.
- Read-only lookup QA can verify exact lookup, code lookup, source-filter isolation, deterministic behavior, unknown unmapped behavior, and ambiguous manual-review behavior.
- Public reports contain aggregate counts and safe status fields only.

Current non-capabilities:
- No runtime B07 integration is active.
- No automatic terminology-backed writes are enabled.
- No clinical recommendation, dosing guidance, or DDI status update is generated.
- No OCR, extraction, routing, confidence, or safety-gate behavior is changed.

## TERM-02 Status

- Commit: `e2ce45f`
- Imported systems: `rxnorm`, `loinc`
- rxnorm rows imported: `244529`
- loinc rows imported: `109325`
- Total local concepts: `353854`
- External API used: `False`
- Private terminology source files and local DB/index files remain uncommitted.

## TERM-03 Status

- Commit: `906f526`
- Conclusion: `cka_term03_local_terminology_qa_ready`
- Store systems detected: `loinc`, `rxnorm`
- QA cases: `10 total`, `9 passed`, `0 failed`, `1 skipped`
- Unknown lookup remains unmapped.
- Ambiguous lookup remains manual-review/ambiguous.
- Determinism passed.
- Source filter isolation passed.
- Code lookup passed.
- Synonym/alias lookup skipped because TERM-02 did not populate synonym rows.

## Proposed Integration Architecture

Future integration should use a read-only terminology lookup adapter between the local terminology store and any clinical coding or annotation workflow.

The adapter boundary should:
- Open the local terminology store in read-only mode.
- Accept normalized query text and optional source-system filters.
- Return only lookup status, safe match metadata, confidence/status labels, and reason codes.
- Preserve unknown input as `unmapped`.
- Preserve ambiguous input as `ambiguous` and require manual review.
- Avoid writing into the clinical knowledge base by default.
- Avoid returning clinical advice, dosing guidance, or DDI decisions.

Recommended call path for future work:
1. UI or diagnostic tool submits a local term query.
2. Read-only terminology adapter queries the private local store.
3. Adapter returns safe lookup result.
4. Caller displays or records a hypothesis-only annotation if explicitly enabled.
5. Existing safety gates and review boundaries remain authoritative.

## Required Feature Flags

Future activation must require explicit flags. Suggested names:
- `MEDAI_TERMINOLOGY_LOOKUP_ENABLED=false` by default
- `MEDAI_TERMINOLOGY_READ_ONLY=true` by default
- `MEDAI_TERMINOLOGY_UI_PANEL_ENABLED=false` by default
- `MEDAI_TERMINOLOGY_HYPOTHESIS_ANNOTATION_ENABLED=false` by default
- `MEDAI_B07_TERMINOLOGY_OPT_IN=false` by default
- `MEDAI_TERMINOLOGY_ALLOW_WRITES=false` by default

No future block should enable automatic writes or B07 integration without an explicit approval block.

## Allowed Future Behavior

Allowed only behind explicit opt-in flags:
- Read-only local lookup from synthetic fixtures.
- Read-only local lookup from the private TERM-02 store.
- UI-only lookup panel that displays safe lookup status and safe match metadata.
- Hypothesis-only coding annotations that leave active clinical facts unchanged.
- Regression reports with aggregate metrics and safe identifiers.

## Forbidden Future Behavior

Future blocks must not:
- Promote terminology lookup results to accepted clinical facts.
- Clear DDI status.
- Generate clinical recommendations.
- Generate medication dosing or prescribing advice.
- Resolve ambiguous terminology silently.
- Invent a code for unknown terms.
- Write source terminology rows, raw paths, private acknowledgments, or license text to public reports.
- Send terminology content or clinical content to external APIs.
- Stage or commit local terminology source files or DB/index files.
- Change OCR, extraction, confidence, routing, or safety-gate behavior.

## B07 Non-Promotion Boundary

B07 may not consume terminology lookup as an authority source until a future explicitly approved B07 opt-in integration block. Even then, lookup results must remain hypothesis-only unless a separate clinical governance block authorizes stronger behavior.

Required B07 invariants:
- Unknown terms remain unmapped.
- Ambiguous terms remain manual-review/ambiguous.
- Lookup result must not promote a hypothesis.
- Lookup result must not mark a clinical fact as resolved.
- Lookup result must not modify DDI status.

## DDI Non-Clearing Boundary

Terminology lookup can identify possible code matches but cannot decide interaction safety. DDI status must not be cleared or downgraded by terminology lookup alone.

## Regression Gates Before Integration

Before any runtime integration, these must pass:
- TERM-03 local terminology QA validation.
- TERM-02 controlled local import validation.
- TERM-01H terminology safety red-team validation.
- Final CKA MVP release validation.
- A future synthetic adapter validation.
- A future private-store read-only adapter validation.
- Public report privacy checker.
- Git staging guard for private terminology files and DB/index files.

## Rollback / Disable Plan

Every future terminology integration block must keep a flag-disabled path that preserves current behavior. Disabling terminology flags must:
- Prevent local terminology lookup from affecting runtime results.
- Prevent terminology-backed annotations from being written.
- Preserve existing B07 behavior.
- Preserve existing clinical safety gates.
- Keep TERM-03 QA available as an offline diagnostic.

## Recommended Future Blocks

1. TERM-05 synthetic read-only terminology adapter
2. TERM-06 private-store read-only adapter validation
3. TERM-07 UI-only terminology lookup panel
4. TERM-08 hypothesis-only coding annotation pilot
5. B07-TERM opt-in integration, only after explicit approval
