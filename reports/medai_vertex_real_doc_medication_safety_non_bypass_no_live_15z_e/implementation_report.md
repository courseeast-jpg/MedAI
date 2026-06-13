# MEDAI-VERTEX-REAL-DOC-MEDICATION-SAFETY-NON-BYPASS-NO-LIVE-15Z-E

- medication_safety_non_bypass_created: `True`
- medication_cases_total: `17`
- medication_cases_passed: `17`
- medication_facts_case_count: `16`
- medication_candidate_facts_total: `16`
- medication_safety_proof_created_count: `14`
- medication_safety_proof_required_block_count: `2`
- forbidden_medical_decision_block_count: `5`
- ddi_decision_blocked: `True`
- contraindication_decision_blocked: `True`
- dosage_advice_blocked: `True`
- treatment_advice_blocked: `True`
- diagnosis_output_blocked: `True`
- future_review_package_created_count: `6`
- future_review_only_count: `6`
- real_doc_live_allowed_count: `0`
- explicit_live_call_request_blocked: `True`
- active_write_blocked: `True`
- auto_accept_blocked: `True`
- ddi_decision_made_count: `0`
- contraindication_decision_made_count: `0`
- dosage_advice_made_count: `0`
- treatment_advice_made_count: `0`
- diagnosis_made_count: `0`
- raw_pii_in_report_count: `0`
- token_map_in_report_count: `0`
- live_call_made: `False`
- external_api_used: `False`
- billing_api_used: `False`
- active_written_count: `0`
- active_mkb_record_created_count: `0`
- auto_accept_true_count: `0`
- privacy_result: `passed`
- billing_check_pending: `True`

## Scope

- Medication mentions are candidate facts only and remain review-bound.
- Forbidden medication decision output requests are blocked by boundary checks only.
- Future packages are review-package-only; no real-doc live authorization is created.
- No provider call, billing API call, active MKB write, production queue mutation, or auto-accept.
