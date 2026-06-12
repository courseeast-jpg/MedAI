# MedAI Capability Boundary Doctrine

## 1. Title and status

- **Title:** MedAI Capability Boundary Doctrine
- **Document ID:** MEDAI-CAPABILITY-BOUNDARY-DOCTRINE-DOC-15H
- **Status:** Active — permanent architecture doctrine and review gate.
- **Applies to:** All current and future MedAI extraction, OCR, rules, AI-assist, review, and MKB work.
- **Authority:** This document is binding for architecture review. A design that conflicts with it must be revised or explicitly justified through the change-control rule in Section 12.

This is a documentation/governance artifact. It changes no runtime behavior on its own.

## 2. Background: why the doctrine exists

Earlier MedAI development followed an **OCR-first** path. Over many iterations the OCR
and deterministic-rules layers were progressively asked to do more and more of the
*meaning* work: infer document type, reconstruct table structure, decide section
boundaries, and ultimately "understand" the clinical document well enough to emit
finished knowledge atoms.

The result was a steady stream of diagnostics, phase patches, and re-tuning that never
converged, because the layers being tuned were structurally incapable of the task being
demanded of them. The lesson was not "the rules had bugs." The lesson was that the
**abstraction boundary itself was wrong**: a deterministic text/layout recovery layer
was being treated as a semantic document-understanding engine.

This doctrine records that lesson so it is not relearned the expensive way, and turns it
into a reusable architecture review gate.

## 3. The architecture-class error

The core mistake was a **capability-boundary error**, not an implementation bug:

> OCR and deterministic rules were treated as if they could become a semantic/layout
> document-understanding engine. This was the wrong abstraction boundary.

Symptoms that this error is recurring:

- "One more rule / one more regex / one more heuristic" is expected to finally produce
  document understanding.
- Layout and section meaning are reconstructed by deterministic pattern matching on raw
  OCR output.
- Extraction quality is chased by tuning OCR or rules instead of by changing which layer
  owns the semantic step.
- Complex documents are decomposed into atoms first, and the human-meaningful package is
  expected to "emerge" from those atoms later.

**Governing principle (exact language):**

> "Do not assign semantic/layout understanding to OCR or deterministic rules."

## 4. Correct component responsibility boundaries

Each layer has one job and is not asked to exceed its capability:

- **OCR** recovers text and layout signals. It is input acquisition, not understanding.
- **Rules** normalize, validate, route, compare, and fail safely. Rules are validators,
  fallbacks, and guards — not the primary meaning engine.
- **AI** reconstructs source-visible review packages. AI is the semantic/layout
  reconstruction layer, and its output is always review-bound.
- **MedAI (the platform)** controls privacy, schema validation, review, safety, and MKB
  writes. MedAI is the control plane; it owns the gates.
- **MKB** stores validated/reviewed knowledge, not raw AI speculation. Nothing reaches
  active MKB without passing schema validation and human review.

Responsibility never flows "upward" into a less-capable layer. OCR/rules must not be made
responsible for semantics; AI output must not be made responsible for its own acceptance.

## 5. Package-first rule

MedAI extraction is **package-first**, not atom-first.

- The unit of extraction is a **source-faithful, operator-visible package** that maps to
  what a human sees in the original document (sections, observations, narrative, labels).
- Atoms (individual observations/records) live **inside** a package; they are never the
  first-class output for complex source documents.
- Packages must be created **while source structure is still available**, not
  reconstructed after structure has already been lost to atomization.
- A package preserves enough source structure and labeling that an operator can compare it
  to the original document quickly.

## 6. Wrong success metrics to reject

The following are explicitly **not** success metrics and must be rejected in review:

- "Records were created."
- "MKB count increased."
- Database/row growth of any kind.
- Number of atoms emitted.
- Internal counters that no operator can see or act on.

> "Records created" or "MKB count increased" is not a success metric.

Success requires source-faithful, operator-visible packages that a human can compare
against the original source quickly.

## 7. Required acceptance test

Every MedAI extraction block must satisfy this acceptance rule (exact language):

> "A MedAI extraction block is not successful because records were created. It is successful only if the operator can compare a source-faithful package against the original document quickly and safely."

Concretely, an extraction block passes only if:

- A reviewer is shown a package whose structure mirrors the source document.
- The reviewer can confirm or reject it against the original quickly.
- Failures are visible to the operator, not buried in internal counters.
- All AI-derived content remains review-bound until a human accepts it.

## 8. Architecture review checklist

Every new MedAI design or extraction block must answer these questions in review:

- What source information must survive?
- Which layer is capable of preserving or reconstructing it?
- Are we creating packages first, or weak atoms first?
- What is the human-visible acceptance test?
- Can failure be seen by the operator, or only in internal counters?
- Does any AI output remain review-bound?
- Is private data protected before any external call?
- Are we measuring useful review content, not database growth?

If any answer reveals semantic work pushed into OCR/rules, atom-first decomposition of a
complex document, count-based success, hidden failure, un-reviewed AI writes, or weakened
privacy, the design is rejected or revised.

## 9. Examples of allowed vs forbidden design patterns

### Forbidden patterns

- OCR/rules as primary semantic document understanding.
- Atomic-record-first extraction for complex source documents.
- Treating count growth as success.
- Hidden internal records without a visible package body.
- Reconstructing packages only after source structure is lost.
- Allowing AI output to bypass review-bound status.
- Weakening privacy gates before external calls.

### Allowed patterns

- Package-first extraction.
- AI-assisted source package drafts.
- Local OCR/text/layout as input acquisition.
- Deterministic rules as validators/fallbacks.
- Privacy gate before external AI.
- Schema validation before review queue.
- Review-bound package write.
- Human accept/reject/defer before active MKB write.

## 10. How this doctrine applies to future AI-extraction blocks

For any future AI-extraction work (e.g. the disabled provider-adapter series and its
successors):

- AI is the semantic/layout reconstruction layer; OCR/rules stay in their lanes from
  Section 4.
- The deliverable is a review-bound, source-faithful **package**, never auto-accepted
  atoms.
- A privacy gate runs before any external call; payloads are redacted text/layout
  summaries only; raw documents/images are never shipped.
- Schema validation runs before anything enters the review queue.
- Real external execution stays gated and disabled-by-policy until separately and
  explicitly enabled; presence of credentials is never sufficient to enable a call.
- The acceptance test from Section 7 — not record counts — decides whether the block
  succeeded.

## 11. Non-goals

This doctrine does **not**:

- Enable, configure, or perform any real external AI call.
- Change OCR, rules, extraction, routing, validation, privacy, or MKB-write behavior.
- Mandate a specific AI provider, model, or vendor.
- Replace per-block specifications, tests, or privacy/safety gates; it constrains them.
- Add patient data, credentials, or source document text to the repository.

## 12. Change control / enforcement rule

- This doctrine is a **review gate**: architecture reviews must check designs against
  Sections 4–9 before approval.
- The governing principle and the acceptance rule (Sections 3 and 7) are **fixed
  language** and must not be silently reworded; changes require an explicit,
  reviewed update to this document referencing the reason.
- Any block that needs to deviate must document the deviation and its justification in the
  block's report and link back to this doctrine.
- Weakening a privacy, schema-validation, review-bound, or MKB-write guard is never an
  allowed "optimization" and must be treated as an architecture regression.
