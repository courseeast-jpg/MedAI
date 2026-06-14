# 16E-C machine review matrix (counts/booleans only)

Approved file (basename only): `2.PNG`
Repaired payload hash: `sha256:ea88594a2ec9a58e`

| Field | Value |
| --- | --- |
| repaired_payload_read_locally | `True` |
| raw_labcorp_remaining | `False` |
| raw_dob_detected | `False` |
| raw_address_detected | `False` |
| raw_phone_or_email_detected | `False` |
| raw_mrn_detected | `False` |
| raw_insurance_or_account_id_detected | `False` |
| raw_accession_or_specimen_id_detected | `False` |
| raw_provider_or_facility_identifier_detected | `False` |
| local_path_detected_in_payload | `False` |
| name_candidate_count | `0` |
| remaining_patient_name_token_count | `32` |
| clinical_table_content_usable | `True` |
| token_map_in_public_report | `False` |
| raw_identifier_leak_count | `0` |
| machine_review_result | `NEEDS_HUMAN_REVIEW` |
| operator_attestation_still_required | `True` |

Counts/booleans only — no raw payload body, no token map, no raw OCR. The
repaired payload and token map remain private, outside git. Operator attestation
is still required; this machine review does not write NO_PHI_ATTESTED.
