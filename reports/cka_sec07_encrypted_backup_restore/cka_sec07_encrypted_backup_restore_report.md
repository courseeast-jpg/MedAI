# CKA-SEC-07 Encrypted Backup / Restore Report

- block_id: CKA-SEC-07
- conclusion: cka_sec07_encrypted_backup_restore_ready

## Tooling

- backup_tool_ready: True
- restore_tool_ready: True
- dry_run_supported: True

## Round-trip

- synthetic_backup_restore_passed: True
- correct_key_restore_passed: True
- wrong_key_restore_failed: True
- checksum_verified: True
- plaintext_absence_verified: True

## Boundaries

- real_data_touched: False
- real_store_modified: False
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

- Case A: [PASS] Baseline SEC-05 + final CKA confirmed
- Case B: [PASS] Synthetic source DB created
- Case C: [PASS] Encrypted backup created
- Case D: [PASS] Backup checksum verified
- Case E: [PASS] Restored backup to temp target
- Case F: [PASS] Correct key opens restored DB
- Case G: [PASS] Wrong key fails on restored DB
- Case H: [PASS] Restored record count matches
- Case I: [PASS] Plaintext absent in backup bytes
- Case J: [PASS] No DB/key/private files staged
- Case K: [PASS] Final CKA validation invocable
- Case L: [PASS] Report safety

## Next recommended action

stop, then consider key rotation plan
