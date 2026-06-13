# 15Z-H operator UAT matrix

| Step | UAT case | Source | Expected | Observed | Status |
| --- | --- | --- | --- | --- | --- |
| 1 | operator_can_find_handoff_doc | docs/operator_handoff/MEDAI_VERTEX_REAL_DOC_READINESS_HANDOFF_15Z.md | handoff doc exists | exists=True | pass |
| 2 | operator_can_identify_real_doc_boundary | docs/operator_handoff/MEDAI_VERTEX_REAL_DOC_READINESS_HANDOFF_15Z.md | real-document route remains blocked | boundary text present | pass |
| 3 | operator_can_identify_gate_inventory | docs/operator_handoff/MEDAI_VERTEX_REAL_DOC_READINESS_HANDOFF_15Z.md | all readiness gates are listed | gates_found=15 | pass |
| 4 | operator_can_run_15z_a_command_no_live | docs/operator_handoff/MEDAI_VERTEX_REAL_DOC_READINESS_HANDOFF_15Z.md | command is listed for local no-live validation | python scripts/run_medai_vertex_real_doc_readiness_gates_no_live_15z_a.py | pass |
| 5 | operator_can_run_15z_b_command_no_live | docs/operator_handoff/MEDAI_VERTEX_REAL_DOC_READINESS_HANDOFF_15Z.md | command is listed for local no-live validation | python scripts/run_medai_vertex_real_doc_pii_stripping_proof_no_live_15z_b.py | pass |
| 6 | operator_can_run_15z_c_command_no_live | docs/operator_handoff/MEDAI_VERTEX_REAL_DOC_READINESS_HANDOFF_15Z.md | command is listed for local no-live validation | python scripts/run_medai_vertex_real_doc_adapter_dry_run_no_live_15z_c.py | pass |
| 7 | operator_can_run_15z_d_command_no_live | docs/operator_handoff/MEDAI_VERTEX_REAL_DOC_READINESS_HANDOFF_15Z.md | command is listed for local no-live validation | python scripts/run_medai_vertex_real_doc_authorization_cost_refusal_gates_no_live_15z_d.py | pass |
| 8 | operator_can_run_15z_e_command_no_live | docs/operator_handoff/MEDAI_VERTEX_REAL_DOC_READINESS_HANDOFF_15Z.md | command is listed for local no-live validation | python scripts/run_medai_vertex_real_doc_medication_safety_non_bypass_no_live_15z_e.py | pass |
| 9 | operator_can_run_15z_f_command_no_live | docs/operator_handoff/MEDAI_VERTEX_REAL_DOC_READINESS_HANDOFF_15Z.md | command is listed for local no-live validation | python scripts/run_medai_vertex_real_doc_readiness_integrated_harness_no_live_15z_f.py | pass |
| 10 | operator_can_verify_integrated_harness_pass | reports/medai_vertex_real_doc_readiness_integrated_harness_no_live_15z_f/summary.json | integrated harness pass metrics present | cases=17/17 | pass |
| 11 | operator_can_verify_future_review_package_only | reports/medai_vertex_real_doc_readiness_integrated_harness_no_live_15z_f/summary.json | future packages are review-only | future_review_only_count=3 | pass |
| 12 | operator_can_verify_blocked_failure_injections | reports/medai_vertex_real_doc_readiness_integrated_harness_no_live_15z_f/summary.json | unsafe cases are blocked | blocked_case_count=14 | pass |
| 13 | operator_can_verify_no_provider_call | reports/medai_vertex_real_doc_readiness_integrated_harness_no_live_15z_f/summary.json | no provider call | live_call_made=False | pass |
| 14 | operator_can_verify_no_billing_api | reports/medai_vertex_real_doc_readiness_integrated_harness_no_live_15z_f/summary.json | no billing API | billing_api_used=False | pass |
| 15 | operator_can_verify_no_active_write | reports/medai_vertex_real_doc_readiness_integrated_harness_no_live_15z_f/summary.json | no active write | active_written_count=0 | pass |
| 16 | operator_can_verify_no_auto_accept | reports/medai_vertex_real_doc_readiness_integrated_harness_no_live_15z_f/summary.json | no auto-accept | auto_accept_true_count=0 | pass |
| 17 | operator_can_verify_no_medical_decision | reports/medai_vertex_real_doc_readiness_integrated_harness_no_live_15z_f/summary.json | no medical decision | medical_decision_made_count=0 | pass |
| 18 | operator_can_verify_no_raw_pii_or_token_map_reports | reports/medai_vertex_real_doc_readiness_integrated_harness_no_live_15z_f/summary.json | no raw private values or token maps in reports | raw_pii=0; token_map=0 | pass |
| 19 | operator_can_identify_stop_conditions | docs/operator_handoff/MEDAI_VERTEX_REAL_DOC_READINESS_HANDOFF_15Z.md | stop conditions are visible | stop_conditions_found=15 | pass |
| 20 | operator_can_identify_next_block_boundary | reports/medai_vertex_real_doc_readiness_operator_handoff_doc_15z_g/summary.json | future boundary and next block are visible | future block boundary present; 15Z-H referenced | pass |

| Metric | Value |
| --- | --- |
| operator_uat_created | `True` |
| operator_uat_steps_total | `20` |
| operator_uat_steps_passed | `20` |
| handoff_doc_found | `True` |
| gate_inventory_verified | `True` |
| operator_commands_verified | `True` |
| integrated_harness_verified | `True` |
| future_review_package_only_verified | `True` |
| blocked_failure_injections_verified | `True` |
| stop_conditions_verified | `True` |
| future_authorization_boundary_verified | `True` |
| real_doc_live_allowed_count | `0` |
| live_call_made | `False` |
| external_api_used | `False` |
| billing_api_used | `False` |
| active_written_count | `0` |
| active_mkb_record_created_count | `0` |
| auto_accept_true_count | `0` |
| medical_decision_made_count | `0` |
| raw_pii_in_report_count | `0` |
| token_map_in_report_count | `0` |
| privacy_result | `passed` |
| billing_check_pending | `True` |
