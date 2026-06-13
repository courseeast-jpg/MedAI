# 15Z-E medication safety matrix

| Case | Candidate facts | Safety proof | Forbidden decision requested | Status | Future review package | Blocked |
| --- | --- | --- | --- | --- | --- | --- |
| medication_mention_candidate_only_no_safety_proof | 1 | False | False | BLOCKED_MEDICATION_SAFETY_PROOF_REQUIRED | False | True |
| medication_mention_candidate_only_with_safety_proof | 1 | True | False | READY_FOR_FUTURE_REVIEW_PACKAGE_ONLY | True | False |
| medication_with_explicit_dose_candidate_only | 1 | True | False | READY_FOR_FUTURE_REVIEW_PACKAGE_ONLY | True | False |
| medication_with_frequency_candidate_only | 1 | True | False | READY_FOR_FUTURE_REVIEW_PACKAGE_ONLY | True | False |
| medication_with_uncertainty_candidate_only | 1 | True | False | READY_FOR_FUTURE_REVIEW_PACKAGE_ONLY | True | False |
| medication_fact_active_write_requested | 1 | True | False | BLOCKED_ACTIVE_WRITE_REQUESTED | False | True |
| medication_fact_auto_accept_requested | 1 | True | False | BLOCKED_AUTO_ACCEPT_REQUESTED | False | True |
| medication_fact_ddi_decision_requested | 1 | True | True | BLOCKED_MEDICAL_DECISION_LOGIC_REQUESTED | False | True |
| medication_fact_contraindication_decision_requested | 1 | True | True | BLOCKED_MEDICAL_DECISION_LOGIC_REQUESTED | False | True |
| medication_fact_dosage_advice_requested | 1 | True | True | BLOCKED_MEDICAL_DECISION_LOGIC_REQUESTED | False | True |
| medication_fact_treatment_advice_requested | 1 | True | True | BLOCKED_MEDICAL_DECISION_LOGIC_REQUESTED | False | True |
| medication_fact_diagnosis_output_requested | 1 | True | True | BLOCKED_MEDICAL_DECISION_LOGIC_REQUESTED | False | True |
| non_medication_fixture | 0 | False | False | MEDICATION_GATE_NOT_REQUIRED_REVIEW_BOUND | False | False |
| medication_with_tokenized_evidence_and_pii_vault_reference | 1 | True | False | READY_FOR_FUTURE_REVIEW_PACKAGE_ONLY | True | False |
| explicit_live_call_requested_even_with_safety_proof | 1 | True | False | BLOCKED_EXPLICIT_LIVE_CALL_REQUESTED | False | True |
| future_authorization_package_with_medication_without_safety_proof | 1 | False | False | BLOCKED_MEDICATION_SAFETY_PROOF_REQUIRED | False | True |
| future_authorization_package_with_medication_with_safety_proof | 1 | True | False | READY_FOR_FUTURE_REVIEW_PACKAGE_ONLY | True | False |

| Metric | Value |
| --- | --- |
| medication_cases_total | `17` |
| medication_candidate_facts_total | `16` |
| forbidden_medical_decision_block_count | `5` |
| future_review_package_created_count | `6` |
| real_doc_live_allowed_count | `0` |
