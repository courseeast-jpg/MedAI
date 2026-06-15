# R22 safety boundary

| Gate | Value |
| --- | --- |
| live_extraction_stopped | `True` |
| provider_model_call_made | `False` |
| gemini_call_made | `False` |
| vertex_call_made | `False` |
| billing_api_call_made | `False` |
| mkb_db_opened | `False` |
| active_mkb_write | `False` |
| future_mkb_import_started | `False` |
| auto_accept_enabled | `False` |
| medical_decision_made | `False` |
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

R22 is a local consolidation layer only. It does not resume live extraction, does not open/import/write MKB, and does not auto-accept any package.
