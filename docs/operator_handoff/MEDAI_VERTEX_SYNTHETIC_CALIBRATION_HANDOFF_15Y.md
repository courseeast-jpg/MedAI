# MedAI Vertex Synthetic Calibration — Operator/Governance Handoff

- **Document ID:** MEDAI-VERTEX-SEMANTIC-CALIBRATION-OPERATOR-HANDOFF-DOC-15Y
- **Status:** Active governance evidence summary (documentation only).
- **Applies to:** The completed synthetic/redacted Vertex semantic calibration stage (15P-C … 15X-R4).

This guide is documentation only. It changes no code, no UI, no decision store, and makes no provider call.

## 1. Purpose

This document summarizes the completed **synthetic/redacted** Vertex semantic calibration stage and its evidence. It is **governance evidence, not real-document clearance**: it records what the calibration proves, what it does not prove, and the gates that must still be met before any real medical document is routed to Vertex.

## 2. Calibration status

- **15X-R4: full synthetic calibration PASS.**
- **15/15 fixtures passed.**
- **12 categories** covered.
- **No active MKB writes.**
- **No auto-accept.**
- **Review-bound only.**

## 3. What 15X-R4 proves

On synthetic/redacted fixtures, with the hardened MedAI semantic contract:

- Vertex can follow the MedAI semantic extraction contract.
- The JSON response schema can be satisfied.
- Verbatim source evidence anchoring can be achieved.
- Declared bilingual label aliases can work deterministically.
- Unknown and uncertainty flags remain explicit.
- Candidate facts stay separated from the source body.
- A token/cost ledger can be generated.

## 4. What 15X-R4 does NOT prove

- It does **not** clear unrestricted real medical documents.
- It does **not** prove PII stripping on real documents.
- It does **not** prove OCR-to-Vertex routing safety.
- It does **not** authorize active MKB writes.
- It does **not** authorize auto-promotion or auto-accept.
- It does **not** prove medication safety gate behavior on real medication writes.
- It does **not** make MedAI a medical decision system.

## 5. Final synthetic calibration metrics

- fixture_count = **15**
- live_call_count = **15**
- schema_validation_pass_count = **15**
- verbatim_evidence_anchor_pass_count = **15**
- hallucinated_field_count = **0**
- review_required_count = **15**
- active_written_count = **0**
- active_mkb_record_created_count = **0**
- auto_accept = **false**
- total_token_count_all_calls = **7334**
- estimated_total_cost_usd_all_calls = **$0.0014486**
- billing_check_pending = **true**

## 6. Category coverage (12)

- portal_result_cards
- cytology_pathology_narrative
- urinalysis_table_like_lab
- mixed_narrative_numeric_result
- short_clinical_note_negation
- short_clinical_note_uncertainty
- medication_mention_no_ddi
- bilingual_cyrillic_snippet
- sparse_low_information_result
- multi_section_explicit_unknowns
- abnormal_numeric_with_units
- normal_numeric_with_units

## 7. Guardrails that held

- Dedicated live gate only (`MEDAI_VERTEX_CALIBRATION_BATCH_SYNTHETIC_LIVE_ALLOWED=YES`).
- One run only.
- ≤ 20 calls.
- One call per fixture.
- No retries.
- Stop on first failure.
- POST body restricted to `{contents, generationConfig}`.
- Synthetic/redacted fixtures only.
- No credentials or auth artifacts in reports.
- No active write / no auto-accept.
- Review-required on all outputs.

## 8. Contract hardenings created during calibration

- **15X-R1 (evidence anchoring):** `evidence_text` must be a verbatim source span; paraphrase is rejected; no embeddings / LLM judge / fuzzy evidence matching.
- **15X-R3 (label aliases):** deterministic, explicitly-declared label aliases only; undeclared label drift rejected; section mismatch rejected; no embeddings / LLM judge / fuzzy label matching.

## 9. Operator interpretation

- PASS means **"safe to continue to readiness design."**
- PASS does **not** mean **"safe to process real documents."**
- Any future real-document route must remain **review-bound** until separately authorized.
- Any active-write path must require **separate approval and tests**.

## 10. Required gates before real-document routing

These are explicit FUTURE gates (not satisfied by 15X-R4):

- Real-doc privacy stripping proof.
- PII vault isolation proof.
- No raw private document leakage in reports/logs.
- Synthetic-to-real adapter dry run.
- Redacted real-like fixture replay.
- Operator review queue handoff.
- Real-doc refusal path.
- Billing/cost cap check.
- Human authorization gate for any real live call.
- No active MKB write gate.
- No auto-accept gate.
- Medication safety gate non-bypass proof if medication facts appear.

## 11. Recommended next implementation stage

- **MEDAI-VERTEX-REAL-DOC-READINESS-GATES-NO-LIVE-15Z**
- No-live only.
- Prepare privacy/readiness gates.
- Do **not** send real documents to Vertex yet.

## 12. Evidence chain

| Block | Scope | Result |
| --- | --- | --- |
| 15P-C | Portal result-card live Vertex comparison | PASS |
| 15P-D | Remaining 3 synthetic families live comparison | PASS |
| 15Q | Operator review surface | PASS |
| 15R | Review-bound decision persistence | PASS |
| 15S | Read-only audit/export | PASS |
| 15T | Operator panel | PASS |
| 15U | Tab wiring | PASS |
| 15V | Operator UAT | PASS |
| 15W | Operator handoff/governance doc | PASS |
| 15X | Initial broader live calibration | FAIL (evidence paraphrase — correct stop) |
| 15X-R1 | No-live evidence-anchor hardening | PASS |
| 15X-R2 | Live rerun | FAIL (bilingual label normalization — correct stop) |
| 15X-R3 | No-live bilingual label alias calibration | PASS |
| 15X-R4 | Full synthetic/redacted live calibration batch | PASS |

Arc: PASS chain (15P-C…15W) → FAIL (15X) → PASS (15X-R1) → FAIL (15X-R2) → PASS (15X-R3) → **PASS (15X-R4)**. Both live failures were correct, contained stops that hardened the contract.

## 13. Troubleshooting / stop conditions

If future work observes any of the following, **stop and treat as a safety failure** (do not work around it):

- real PII in an outbound payload
- raw private content in reports
- a provider call without the dedicated gate
- an active MKB write
- `auto_accept=true`
- missing `review_required`
- non-verbatim evidence accepted
- undeclared label drift accepted
- medication safety bypass
