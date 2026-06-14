# MEDAI-REAL-DOC-REPAIRED-PAYLOAD-MACHINE-REVIEW-LOCAL-ONLY-16E-C

## Machine review result: **NEEDS_HUMAN_REVIEW** (safety: passed, privacy: needs_human_review)

## Result

- Local heuristic machine review of the repaired tokenized payload for one approved
  document. No provider call, no network, no billing, no live gate activation.
- Produced sanitized counts and pass/fail flags only; the repaired payload, token
  map, and raw OCR are never committed.
- This machine review does not constitute human attestation and never writes
  NO_PHI_ATTESTED. 16D is not started.

## Metrics

- local_only: `True`
- approved_basename: `2.PNG`
- repaired_payload_read_locally: `True`
- raw_labcorp_remaining: `False`
- raw_patient_identifier_detected: `False`
- raw_dob_detected: `False`
- raw_address_detected: `False`
- raw_phone_or_email_detected: `False`
- raw_mrn_detected: `False`
- raw_insurance_or_account_id_detected: `False`
- raw_accession_or_specimen_id_detected: `False`
- raw_provider_or_facility_identifier_detected: `False`
- local_path_detected_in_payload: `False`
- name_candidate_count: `0`
- remaining_patient_name_token_count: `32`
- clinical_terms_present_count: `19`
- clinical_table_content_usable: `True`
- tokenized_payload_committed: `False`
- raw_ocr_committed: `False`
- token_map_committed: `False`
- public_report_phi_leak_count: `0`
- provider_call_made: `False`
- vertex_live_execution: `False`
- billing_api_call_made: `False`
- future_live_gate_environment_active: `False`
- sixteen_d_retry_started: `False`
- machine_review_result: `NEEDS_HUMAN_REVIEW`
- operator_attestation_still_required: `True`
- privacy_result: `needs_human_review`
- safety_result: `passed`

## Notes

- `auto_pass_withheld_pending_human_attestation`

## Recommended next (NO live retry started automatically)

- Human/operator reads this report, reviews the private repaired payload line by
  line, and completes the No-PHI attestation. A future 16D retry requires explicit
  new authorization and is not started by this block.
