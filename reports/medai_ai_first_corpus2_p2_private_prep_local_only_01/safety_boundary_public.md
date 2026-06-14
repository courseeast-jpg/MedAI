# Corpus 2 / P2 private prep — safety boundary

| Gate | Value |
| --- | --- |
| provider_model_call_made | `False` |
| vertex_call_made | `False` |
| gemini_call_made | `False` |
| billing_api_call_made | `False` |
| live_gate_set | `False` |
| mkb_db_opened | `False` |
| active_mkb_write | `False` |
| auto_accept_enabled | `False` |
| medical_decision_made | `False` |
| tokenized_payloads_private_only | `True` |
| token_maps_private_only | `True` |
| private_artifacts_committed | `False` |
| raw_ocr_committed | `False` |
| tokenized_payloads_committed | `False` |
| token_maps_committed | `False` |
| pi_values_committed | `False` |
| credentials_or_tokens_committed | `False` |
| public_report_phi_leak_count | `0` |
| private_path_leaks_after | `0` |
| secret_leaks_after | `0` |
| privacy_result | `passed` |
| safety_result | `passed` |

Local-only preparation. No provider/billing/model call, no live gate, no MKB. PI vault, raw extraction/OCR, token maps, tokenized payloads, and the outbound package are private (outside the repo) and never committed. Public reports carry only counts, short/grouped hashes, basenames, family labels, validation status, and redacted private-path labels.
