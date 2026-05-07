# CKA-SEC-06 Operator Key Rotation Report

- block_id: CKA-SEC-06
- conclusion: cka_sec06_key_rotation_plan_ready

## Tooling

- key_rotation_tool_ready: True
- synthetic_rotation_rehearsal_passed: True
- backup_before_rotation_required: True
- backup_checksum_verified: True
- new_key_open_after_rotation_passed: True
- old_key_rejected_after_rotation: True
- record_count_preserved: True
- rollback_restore_verified: True
- plaintext_absence_verified: True

## Key handling

- command_line_keys_rejected: True
- empty_key_refused: True
- key_mismatch_refused: True
- same_old_new_key_refused: True

## Boundaries

- real_rotation_blocked_by_default: True
- real_rotation_performed: False
- real_store_touched: False
- real_data_touched: False
- db_file_staged: False
- key_stored_in_repo: False
- encryption_key_logged: False
- external_api_used: False
- raw_phi_logged_in_public_reports: False
- private_filename_path_leaks: 0
- secret_leaks: 0
- clinical_recommendations_generated: False
- prescription_dosing_advice_generated: False
- production_ocr_changed: False
- production_extractor_changed: False
- safety_gate_changed: False
- frozen_hitl_release_reopened: False

## Cases

- Case A: [PASS] Baseline SEC-07/SEC-05/final-CKA confirmed
- Case B: [PASS] CLI rejects key args
- Case C: [PASS] Key mismatch refused
- Case D: [PASS] Empty key refused
- Case E: [PASS] Same old/new key refused
- Case F: [PASS] Synthetic rotation rehearsal
- Case G: [PASS] Rollback restore verified
- Case H: [PASS] Real rotation blocked by default
- Case I: [PASS] No DB/key/private files staged
- Case J: [PASS] Report safety

## Next recommended action

stop; key rotation execution on a real store requires separate explicit operator approval
