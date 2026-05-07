# CKA-TERM-02 Execution Blueprint

CKA-TERM-02 is the future controlled local terminology import. It may start only after the operator supplies licensed terminology files locally and creates a private `LICENSE_ACK_PRIVATE.json` acknowledgment. This blueprint is planning only; TERM-02 is not run here.

## Completed Preparation Blocks

- TERM-01: local terminology readiness and license gate.
- TERM-01A: intake folders, classifier, safe copy, safe extract, and readiness check.
- TERM-01B: dry-run import planner, limits, row caps, and checkpoint model.
- TERM-01C: synthetic transaction executor and rollback scaffold.
- TERM-01D: synthetic QA golden lookup harness.
- TERM-01E: operator readiness visibility.
- TERM-01F: manual return kit and TERM-02 preflight gate.
- TERM-01G: synthetic scale, performance bucket, and resume harness.
- TERM-01H: safety red-team and privacy regression pack.

## TERM-02 Phases

1. Preflight: run the TERM-02 preflight gate and stop unless it allows execution.
2. Inventory: inventory local terminology files using safe system labels and counts only.
3. License confirmation: confirm `operator_acknowledged=true` and acknowledged systems cover systems with files.
4. Dry-run plan: generate the TERM-01B plan and review row caps, chunks, blocked systems, and missing systems.
5. Capped import: execute a controlled local import using row caps and checkpoints. Real import remains blocked until TERM-02 explicitly enables it.
6. QA harness: run TERM-01D golden lookup checks against the resulting local store.
7. B07 boundary check: verify coding remains opt-in, unknowns remain unmapped, hypothesis tier is not promoted, and DDI state is not cleared.
8. Privacy report: run report privacy checks and verify no raw private content appears in public reports.
9. Commit and tag policy is defined so source code and public reports are the only commit candidates after all gates pass.

## Execution Invariants

- No external terminology API is used.
- No connector or cloud API is activated.
- No clinical logic is changed.
- No B07 default behavior is changed.
- No unknown term receives an invented code.
- Ambiguous lookup is flagged instead of silently resolved.
- Public reports use safe source IDs, counts, and reason codes only.

## Version Control Rules

- Commit source code and PHI-safe public reports only.
- Never commit `terminology_data/`.
- Never commit `data/terminology/`.
- Never commit `LICENSE_ACK_PRIVATE.json`.
- Never commit database, key, private, PDF, image, or archive artifacts.
- Never commit copied vendor files or generated terminology indexes.

## TERM-02 May Start Only If

- The preflight gate allows execution.
- At least one supported system is import-ready.
- License acknowledgment covers every system with files.
- No terminology files or terminology database files are staged.
- TERM-01H red-team validation still passes.
- Final CKA validation still passes.
