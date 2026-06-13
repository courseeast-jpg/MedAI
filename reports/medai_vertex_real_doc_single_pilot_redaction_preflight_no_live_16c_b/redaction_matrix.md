# 16C-B redaction matrix

| Synthetic case | Findings | Tokens | Classes |
| --- | --- | --- | --- |
| synthetic_case_1_demographics | 5 | 5 | ADDRESS, DOB, EMAIL, PATIENT_NAME, PHONE |
| synthetic_case_2_identifiers | 4 | 4 | ACCESSION, ACCOUNT_ID, INSURANCE_ID, MRN |
| synthetic_case_3_care_team | 3 | 3 | DATE, FACILITY, PROVIDER |
| synthetic_case_4_file_metadata | 4 | 4 | FILENAME, METADATA, OCR_ARTIFACT, PATH |
| synthetic_case_5_freetext_and_rare | 2 | 2 | FREETEXT_ID, RARE_COMBO |
| synthetic_case_6_mixed | 6 | 6 | ACCESSION, DOB, FACILITY, MRN, PATIENT_NAME, PROVIDER |

| Field | Value |
| --- | --- |
| no_live | `True` |
| synthetic_only | `True` |
| redaction_preflight_executed | `True` |
| tokenization_preflight_executed | `True` |
| synthetic_fixture_count | `6` |
| raw_identifier_leak_count | `0` |
| token_map_written_to_public_report | `False` |
| outbound_payload_contains_raw_identifier | `False` |
| outbound_payload_tokenized | `True` |
| future_live_gate_set | `False` |
| future_live_gate_environment_active | `False` |
| future_16d_not_started | `True` |
| privacy_result | `passed` |
| safety_result | `passed` |

Token classes covered: PATIENT_NAME, DOB, DATE, ADDRESS, PHONE, EMAIL, MRN, INSURANCE_ID, ACCOUNT_ID, PROVIDER, FACILITY, ACCESSION, FILENAME, PATH, METADATA, OCR_ARTIFACT, FREETEXT_ID, RARE_COMBO.
Tokens only ever appear as `[CLASS_n]`. Raw synthetic identifiers and the token
map never appear in the outbound payload preview or in these reports.
