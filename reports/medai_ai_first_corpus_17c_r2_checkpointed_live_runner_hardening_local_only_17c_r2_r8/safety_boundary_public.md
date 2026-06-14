# 17C-R2-R8 safety boundary

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
| simulation_provider_call_made | `False` |
| private_checkpoint_committed | `False` |
| private_responses_committed | `False` |
| parsed_private_responses_committed | `False` |
| tokenized_payloads_written_to_repo | `False` |
| raw_ocr_written_to_repo | `False` |
| token_maps_written_to_repo | `False` |
| private_identifier_values_written_to_repo | `False` |
| credential_or_token_written_to_repo | `False` |
| public_report_phi_leak_count | `0` |
| privacy_result | `passed` |
| safety_result | `passed` |

All durable checkpoint state and preserved evidence live OUTSIDE the repo and are never committed. Public reports carry only status, counts, hashes, and failure categories — never a response body, tokenized payload, token map, private identifier, or credential.
