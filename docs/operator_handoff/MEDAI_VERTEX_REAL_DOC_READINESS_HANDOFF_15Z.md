# MEDAI Vertex Real Document Readiness Handoff 15Z

## 1. Purpose

This is the no-live real-document readiness gate handoff for the MedAI Vertex path. It summarizes the completed 15Z-A through 15Z-F readiness blocks and gives operators a single place to review the current gate state.

This document does not authorize real-document Vertex routing. It is documentation for a blocked-by-default readiness chain.

## 2. Current status

- 15Z-A through 15Z-F PASS.
- Integrated harness PASS.
- `real_doc_live_allowed_count=0`.
- `live_call_made=false`.
- `external_api_used=false`.
- `billing_api_used=false`.
- `active_written_count=0`.
- `auto_accept_true_count=0`.
- `privacy_result=passed`.

## 3. Gate inventory

- `real_doc_external_routing_default_block`
- `pii_stripping_proof_required`
- `pii_vault_isolation_required`
- `no_raw_private_payload_in_reports_required`
- `synthetic_to_real_adapter_dry_run_required`
- `redacted_real_like_fixture_replay_required`
- `operator_review_queue_handoff_required`
- `human_authorization_required_for_any_real_live_call`
- `no_active_mkb_write_required`
- `no_auto_accept_required`
- `medication_safety_non_bypass_required_if_medication_facts_present`
- `billing_cost_cap_ack_required`
- `dedicated_future_real_doc_live_gate_required`
- `real_doc_refusal_path_required`
- `no_medical_decision_logic_required`

## 4. What 15Z-A proves

15Z-A proves the default-deny real-document readiness framework. Real/private provenance and unknown provenance are blocked. The only possible outcome for future live routing is future authorization only; there is no live authorization in the block.

## 5. What 15Z-B proves

15Z-B proves PII-like values are tokenized before any would-be provider payload. The vault map remains isolated. Raw PII does not appear in outbound payloads or reports. The token map does not appear in payloads or reports.

## 6. What 15Z-C proves

15Z-C proves a would-be Vertex request shape can be assembled offline. The top-level keys are exactly `contents` and `generationConfig`. Forbidden MedAI/provider metadata remains local. Request fingerprinting works. Review handoff is report-only.

## 7. What 15Z-D proves

15Z-D proves human authorization and billing acknowledgement are modeled in no-live mode. Missing authorization or billing acknowledgement blocks. Explicit live-call requests block. The cost cap is local/no-live only. No billing API is used.

## 8. What 15Z-E proves

15Z-E proves medication mentions remain candidate facts only. Medication safety proof is required when medication facts are present. DDI, contraindication, dosage, treatment, and diagnosis decisions are blocked. No medication decision logic is introduced.

## 9. What 15Z-F proves

15Z-F proves the gates compose end-to-end. Clean synthetic/redacted-real-like cases can create future operator review packages only. Failure injections are blocked. No live real-document routing is authorized.

Integrated 15Z-F metrics:

- `integrated_readiness_harness_created=true`
- `integrated_cases_total=17`
- `integrated_cases_passed=17`
- `future_operator_review_package_created_count=3`
- `future_operator_review_only_count=3`
- `blocked_case_count=14`
- `refusal_records_created_count=14`
- `pii_redaction_pass_count=15`
- `vault_isolation_pass_count=16`
- `request_shape_valid_count=15`
- `review_handoff_created_count=7`
- `authorization_intent_present_count=6`
- `billing_ack_present_count=6`
- `medication_safety_proof_present_count=3`
- `explicit_live_call_request_blocked=true`
- `active_write_blocked=true`
- `auto_accept_blocked=true`
- `medication_safety_non_bypass_enforced=true`
- `medical_decision_blocked=true`
- `real_doc_live_allowed_count=0`
- `raw_pii_in_package_count=0`
- `raw_pii_in_report_count=0`
- `token_map_in_package_count=0`
- `token_map_in_report_count=0`
- `medical_decision_made_count=0`
- `live_call_made=false`
- `external_api_used=false`
- `billing_api_used=false`
- `active_written_count=0`
- `active_mkb_record_created_count=0`
- `auto_accept_true_count=0`
- `privacy_result=passed`
- `billing_check_pending=true`

## 10. What this does not prove

- It does not prove production real-document routing.
- It does not authorize sending real documents to Vertex.
- It does not prove performance on actual private records.
- It does not authorize active MKB writes.
- It does not authorize auto-accept.
- It does not create a medical decision system.
- It does not validate OCR routing to Vertex on real documents.
- It does not replace operator review.

## 11. Operator commands

```powershell
python scripts/run_medai_vertex_real_doc_readiness_gates_no_live_15z_a.py
python scripts/run_medai_vertex_real_doc_pii_stripping_proof_no_live_15z_b.py
python scripts/run_medai_vertex_real_doc_adapter_dry_run_no_live_15z_c.py
python scripts/run_medai_vertex_real_doc_authorization_cost_refusal_gates_no_live_15z_d.py
python scripts/run_medai_vertex_real_doc_medication_safety_non_bypass_no_live_15z_e.py
python scripts/run_medai_vertex_real_doc_readiness_integrated_harness_no_live_15z_f.py
```

The listed commands are local no-live validation commands. Blocked outcomes: provider_call, billing_api_call, active_write, production_review_queue_mutation, auto_accept, medical_decision_output.

## 12. Stop conditions

Stop and treat the run as a safety failure if any of these conditions appear:

- live provider usage
- billing API usage
- real-document live authorization
- active MKB write
- production review queue mutation
- `auto_accept=true`
- raw PII in payload or report
- token map leakage
- unknown provenance allowed
- real/private marker allowed
- forbidden request metadata accepted
- invalid `generationConfig` accepted
- medication decision output
- medical advice
- missing `review_required`

## 13. Future authorization boundary

Any future real-document live call requires a new explicit block. That future block must have its own live gate, use only redacted/tokenized payloads, use a bounded call count, and stop on first failure.

That future block must remain review-bound. It must not write active MKB, must not auto-accept, and must not make medical decisions.

Real-document Vertex routing remains NOT authorized after this block.

## 14. Recommended next block

Recommended next block: `MEDAI-VERTEX-REAL-DOC-READINESS-OPERATOR-UAT-NO-LIVE-15Z-H`.

Purpose: operator UAT over the no-live readiness handoff and integrated harness, still no real documents and no provider calls.
