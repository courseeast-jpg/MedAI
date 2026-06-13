# 15Z-A real-document readiness gate matrix

| Case | Classification | Status | Blocked | live_call_allowed |
| --- | --- | --- | --- | --- |
| synthetic_calibration_fixture | synthetic_calibration_fixture | `BLOCKED_NO_LIVE_DRY_RUN_PENDING_GATES` | `True` | `False` |
| redacted_real_like_fixture | redacted_real_like_fixture | `BLOCKED_NO_LIVE_DRY_RUN_PENDING_GATES` | `True` | `False` |
| real_private_document_marker | real_private_document | `BLOCKED_REAL_DOC_ROUTING_DEFAULT_DENY` | `True` | `False` |
| unknown_provenance_payload | unknown_provenance_payload | `BLOCKED_REAL_DOC_ROUTING_DEFAULT_DENY` | `True` | `False` |
| payload_with_pii_marker | real_private_document | `BLOCKED_REAL_DOC_ROUTING_DEFAULT_DENY` | `True` | `False` |
| payload_with_raw_pdf_marker | real_private_document | `BLOCKED_REAL_DOC_ROUTING_DEFAULT_DENY` | `True` | `False` |
| payload_with_ocr_private_marker | real_private_document | `BLOCKED_REAL_DOC_ROUTING_DEFAULT_DENY` | `True` | `False` |
| medication_fact_without_safety_gate | synthetic_calibration_fixture | `BLOCKED_NO_LIVE_DRY_RUN_PENDING_GATES` | `True` | `False` |
| active_write_requested | synthetic_calibration_fixture | `BLOCKED_NO_LIVE_DRY_RUN_PENDING_GATES` | `True` | `False` |
| auto_accept_requested | synthetic_calibration_fixture | `BLOCKED_NO_LIVE_DRY_RUN_PENDING_GATES` | `True` | `False` |
| missing_human_authorization | synthetic_calibration_fixture | `BLOCKED_NO_LIVE_DRY_RUN_PENDING_GATES` | `True` | `False` |
| missing_billing_ack | synthetic_calibration_fixture | `BLOCKED_NO_LIVE_DRY_RUN_PENDING_GATES` | `True` | `False` |
| all_future_gates_simulated_pass_no_live | redacted_real_like_fixture | `READY_FOR_FUTURE_AUTHORIZATION_ONLY` | `True` | `False` |

| Metric | Value |
| --- | --- |
| readiness_cases_total | `13` |
| blocked_case_count | `13` |
| no_live_replay_allowed_count | `8` |
| real_doc_live_allowed_count | `0` |
| future_authorization_only_count | `1` |
| real_private_document_blocked | `True` |
| unknown_provenance_blocked | `True` |
| pii_marker_blocked | `True` |
| raw_payload_marker_blocked | `True` |
| ocr_private_marker_blocked | `True` |
| medication_safety_non_bypass_enforced | `True` |
| human_authorization_required | `True` |
| billing_ack_required | `True` |
| active_write_blocked | `True` |
| auto_accept_blocked | `True` |
| sanitized_reports_only | `True` |
| raw_payload_in_report_count | `0` |
| live_call_made | `False` |
| external_api_used | `False` |
| active_written_count | `0` |
| active_mkb_record_created_count | `0` |
| auto_accept_true_count | `0` |
| privacy_result | `passed` |
| billing_check_pending | `True` |

No-live readiness gates: real-doc routing default-deny; even all-gates-pass yields future-authorization-only.
