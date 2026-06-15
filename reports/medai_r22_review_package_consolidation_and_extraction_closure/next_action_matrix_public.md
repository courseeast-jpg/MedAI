# R22 next action matrix

| Package type | Allowed next actions | Disallowed actions |
| --- | --- | --- |
| full_schema | inspect_source_privately; request_manual_operator_review; accept_review_package_for_future_mkb_import_candidate; defer | auto_accept_to_mkb; write_to_mkb_now; use_as_medical_decision |
| minimal_review_bound | inspect_source_privately; request_manual_operator_review; defer | auto_accept_to_mkb; write_to_mkb_now; use_as_medical_decision |
| review_only_finalized | inspect_source_privately; keep_review_only; request_manual_operator_review; defer | send_review_only_records_to_provider_again; auto_accept_to_mkb; write_to_mkb_now; use_as_medical_decision |
| non_sendable_excluded | exclude_from_medical_extraction; defer | send_excluded_rtf_signal_containers; auto_accept_to_mkb; write_to_mkb_now; use_as_medical_decision |

Allowed actions: `inspect_source_privately, accept_review_package_for_future_mkb_import_candidate, keep_review_only, exclude_from_medical_extraction, request_manual_operator_review, defer`
Disallowed actions: `auto_accept_to_mkb, write_to_mkb_now, use_as_medical_decision, send_review_only_records_to_provider_again, send_excluded_rtf_signal_containers`
