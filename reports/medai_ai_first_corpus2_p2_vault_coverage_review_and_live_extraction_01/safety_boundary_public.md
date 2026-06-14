# Corpus 2 / P2 vault review + live — safety boundary

| Gate | Value |
| --- | --- |
| provider_model_call_made | `False` |
| gemini_call_made | `False` |
| mkb_db_opened | `False` |
| active_mkb_write | `False` |
| auto_accept_enabled | `False` |
| medical_decision_made | `False` |
| future_mkb_import_started | `False` |
| corpus1_touched | `False` |
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

Live gate is set only inside an authorized provider call. PI values, raw text, and raw AI responses are never printed or committed. Private vault, token maps, tokenized payloads, raw extraction, responses, checkpoint, and evidence stay outside the repo. Corpus 1 is not read or modified by this block.
