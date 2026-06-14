# Corpus 1 R16 flash-rescue — safety boundary

| Gate | Value |
| --- | --- |
| provider_model_call_made | `True` |
| gemini_call_made | `True` |
| mkb_db_opened | `False` |
| active_mkb_write | `False` |
| auto_accept_enabled | `False` |
| medical_decision_made | `False` |
| future_mkb_import_started | `False` |
| completed_docs_preserved | `True` |
| completed_docs_reprocessed | `False` |
| corpus1_main_checkpoint_mutated | `False` |
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

Rescue uses a SEPARATE R16 checkpoint/evidence; the canonical 118-completed checkpoint and Corpus 1 closure reports are not mutated. No MKB, no auto-accept, no medical decision. Raw responses/PI/text never printed or committed.
