# R21 exhaustive residual recovery safety boundary

| Gate | Value |
| --- | --- |
| mkb_db_opened | `False` |
| active_mkb_write | `False` |
| auto_accept_enabled | `False` |
| medical_decision_made | `False` |
| future_mkb_import_started | `False` |
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

All private payloads, provider evidence, checkpoints, and terminal ledgers stay outside the repository. No MKB open/import/write or auto-accept occurs.
