# 16C-A live gate matrix

Dedicated future live gate (named only): `MEDAI_VERTEX_REAL_DOC_SINGLE_PILOT_LIVE_APPROVED` — current required value: unset or false.

| Field | Value |
| --- | --- |
| no_live | `True` |
| provider_call_made | `False` |
| vertex_live_execution | `False` |
| gemini_live_execution | `False` |
| claude_live_execution | `False` |
| openai_live_execution | `False` |
| billing_api_call_made | `False` |
| real_private_document_processed | `False` |
| private_corpus_read | `False` |
| whole_corpus_processed | `False` |
| pdf_or_image_processed | `False` |
| ocr_routing_executed | `False` |
| active_mkb_write | `False` |
| mkb_db_opened | `False` |
| auto_accept_enabled | `False` |
| medical_decision_made | `False` |
| production_queue_mutated | `False` |
| future_live_gate_named | `True` |
| future_live_gate_set | `False` |
| future_live_gate_environment_active | `False` |
| operator_approval_required | `True` |
| cost_cap_required | `True` |
| redaction_preflight_required | `True` |
| one_document_limit_required | `True` |
| one_call_limit_required | `True` |
| stop_on_first_failure_required | `True` |
| future_16d_not_started | `True` |
| sandbox_treated_as_medai_validation | `False` |
| safety_result | `passed` |

The gate is named/speced only and is NOT set in this block. The gate alone is not
enough: operator approval, cost cap, redaction preflight, one-document limit, and
one-call limit must also pass before any future one-call pilot. 16D is not started.
