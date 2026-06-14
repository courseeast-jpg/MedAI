# 17C-R2-R9 safety boundary

| Gate | Value |
| --- | --- |
| provider_model_call_made | `False` |
| vertex_model_call_made | `False` |
| gemini_call_made | `False` |
| claude_call_made | `False` |
| openai_call_made | `False` |
| billing_api_call_made | `False` |
| live_gate_set | `False` |
| live_extraction_started | `False` |
| mkb_db_opened | `False` |
| active_mkb_write | `False` |
| auto_accept_enabled | `False` |
| medical_decision_made | `False` |
| production_queue_mutated | `False` |
| raw_failed_response_committed | `False` |
| raw_failed_response_printed | `False` |
| private_responses_committed | `False` |
| parsed_private_responses_committed | `False` |
| tokenized_payloads_written_to_repo | `False` |
| raw_ocr_written_to_repo | `False` |
| token_maps_written_to_repo | `False` |
| private_identifier_values_written_to_repo | `False` |
| credential_or_token_written_to_repo | `False` |
| private_filename_path_leaks_after | `0` |
| secret_leaks_after | `0` |
| public_report_phi_leak_count | `0` |
| privacy_result | `passed` |
| safety_result | `passed` |

Triage used PRIVATE preserved evidence only. No response body, tokenized payload, token map, private identifier, or credential was printed or committed. Public 17C-R2 reports now carry safe path/secret labels.
