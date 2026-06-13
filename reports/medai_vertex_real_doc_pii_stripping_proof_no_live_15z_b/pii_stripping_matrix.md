# 15Z-B PII stripping proof matrix

| Case | Classification | Findings | Tokens | Status | Blocked | live_call_allowed |
| --- | --- | --- | --- | --- | --- | --- |
| redacted_real_like_basic_note_with_name_dob_mrn | redacted_real_like_fixture | 3 | 3 | `READY_FOR_NO_LIVE_REPLAY_ONLY` | `False` | `False` |
| redacted_real_like_lab_report_with_facility_accession_provider | redacted_real_like_fixture | 3 | 3 | `READY_FOR_NO_LIVE_REPLAY_ONLY` | `False` | `False` |
| redacted_real_like_contact_fields_phone_email_address | redacted_real_like_fixture | 3 | 3 | `READY_FOR_NO_LIVE_REPLAY_ONLY` | `False` | `False` |
| multi_section_report_with_repeated_same_identifier | redacted_real_like_fixture | 3 | 2 | `READY_FOR_NO_LIVE_REPLAY_ONLY` | `False` | `False` |
| fixture_with_unredacted_pii_residue_should_block | redacted_real_like_fixture | 2 | 2 | `BLOCKED_UNREDACTED_PII_RESIDUE` | `True` | `False` |
| fixture_with_token_map_in_payload_should_block | redacted_real_like_fixture | 2 | 2 | `BLOCKED_TOKEN_MAP_LEAK_IN_OUTBOUND_PAYLOAD` | `True` | `False` |
| fixture_with_token_map_in_report_should_block | redacted_real_like_fixture | 2 | 2 | `BLOCKED_TOKEN_MAP_LEAK_IN_REPORT` | `True` | `False` |
| unknown_provenance_even_redacted_should_block | unknown_provenance_payload | 2 | 2 | `BLOCKED_PROVENANCE_NOT_REDACTED_REAL_LIKE` | `True` | `False` |
| real_private_marker_even_redacted_should_block | real_private_document | 2 | 2 | `BLOCKED_PROVENANCE_NOT_REDACTED_REAL_LIKE` | `True` | `False` |
| sanitized_redacted_real_like_ready_for_no_live_replay_only | redacted_real_like_fixture | 9 | 9 | `READY_FOR_NO_LIVE_REPLAY_ONLY` | `False` | `False` |

| Metric | Value |
| --- | --- |
| pii_cases_total | `10` |
| pii_cases_passed | `10` |
| pii_findings_total | `31` |
| pii_tokens_total | `30` |
| token_classes_detected_count | `9` |
| deterministic_repeated_token_count | `1` |
| outbound_payloads_created_count | `10` |
| raw_pii_in_outbound_payload_count | `0` |
| raw_pii_in_report_count | `0` |
| token_map_in_outbound_payload_count | `0` |
| token_map_in_report_count | `0` |
| vault_records_created_count | `10` |
| vault_records_isolated_count | `10` |
| readiness_cases_fed_count | `10` |
| no_live_replay_allowed_count | `5` |
| blocked_case_count | `5` |
| real_doc_live_allowed_count | `0` |
| live_call_made | `False` |
| external_api_used | `False` |
| active_written_count | `0` |
| active_mkb_record_created_count | `0` |
| auto_accept_true_count | `0` |
| privacy_result | `passed` |
| billing_check_pending | `True` |

Token classes: tokens only ever appear as `[CLASS_n]`; raw PII-like values and the
token map never appear in outbound payloads or in these reports.
