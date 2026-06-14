# R17 fast same-day flash recovery — safety boundary

| Gate | Value |
| --- | --- |
| provider_model_call_made | `True` |
| gemini_call_made | `True` |
| mkb_db_opened | `False` |
| active_mkb_write | `False` |
| auto_accept_enabled | `False` |
| medical_decision_made | `False` |
| future_mkb_import_started | `False` |
| corpus1_main_checkpoint_mutated | `False` |
| r16_checkpoint_mutated | `False` |
| corpus1_completed_docs_reprocessed | `False` |
| corpus2_completed_docs_reprocessed | `False` |
| corpus2_oversized_docs_sent | `False` |
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

R17 uses its own checkpoints; the Corpus 1 main 118-completed checkpoint and the R16 checkpoint are not mutated. Minimal-schema results are review-bound only and never promoted to MKB. No raw responses/PI/text printed or committed.
