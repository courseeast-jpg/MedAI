# R18 stronger-model fallback — safety boundary

| Gate | Value |
| --- | --- |
| stronger_model_route_available | `True` |
| stronger_model_used | `True` |
| grounding_or_search_used | `False` |
| provider_model_call_made | `True` |
| gemini_call_made | `True` |
| mkb_db_opened | `False` |
| active_mkb_write | `False` |
| auto_accept_enabled | `False` |
| medical_decision_made | `False` |
| future_mkb_import_started | `False` |
| corpus1_main_checkpoint_mutated | `False` |
| r17_checkpoint_mutated | `False` |
| corpus1_completed_docs_reprocessed | `False` |
| corpus2_completed_docs_reprocessed | `False` |
| corpus2_oversized_docs_sent | `False` |
| cost_cap_exceeded | `False` |
| private_artifacts_committed | `False` |
| raw_ai_response_committed | `False` |
| tokenized_payloads_committed | `False` |
| token_maps_committed | `False` |
| pi_values_committed | `False` |
| credentials_or_tokens_committed | `False` |
| public_report_phi_leak_count | `0` |
| private_path_leaks_after | `0` |
| secret_leaks_after | `0` |
| privacy_result | `passed` |
| safety_result | `passed` |

Stronger model runs only through the existing safety-gated adapter on the same privacy-tokenized payloads. R18 uses its own checkpoints; main 118, R17, and corpus2 checkpoints are untouched. Review-bound minimal results never promoted to MKB. No raw responses/PI/text printed or committed.
