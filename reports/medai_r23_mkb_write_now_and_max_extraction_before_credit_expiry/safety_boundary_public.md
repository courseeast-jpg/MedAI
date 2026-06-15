# R23 safety boundary

| Gate | Value |
| --- | --- |
| user_authorized_mkb_write_now | `True` |
| mkb_db_opened_for_write | `False` |
| mkb_write_scope | `unverified_review_required_only` |
| active_verified_mkb_records_written | `0` |
| auto_accept_enabled | `False` |
| medical_decision_made | `False` |
| all_r23_records_review_required | `True` |
| private_artifacts_committed | `False` |
| raw_ai_response_committed | `False` |
| raw_text_committed | `False` |
| tokenized_payloads_committed | `False` |
| token_maps_committed | `False` |
| pi_values_committed | `False` |
| credentials_or_tokens_committed | `False` |
| public_report_phi_leak_count | `0` |
| private_path_leaks_after | `0` |
| secret_leaks_after | `0` |
| privacy_result | `passed` |
| safety_result | `passed` |

R23 writes only unverified, review-required staging records. It does not promote active verified facts, auto-accept, or make medical decisions.
