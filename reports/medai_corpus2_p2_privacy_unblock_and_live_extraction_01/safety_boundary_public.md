# Corpus 2 / P2 privacy-unblock — safety boundary

| Gate | Value |
| --- | --- |
| provider_model_call_made | `True` |
| gemini_call_made | `True` |
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

Over-tokenization placeholders and candidate values stay in the PRIVATE supplemental vault and token maps outside the repo. No PI value, candidate value, raw text, or raw AI response is printed or committed. Corpus 1 untouched.
