# CKA-SEC-04 Encrypted Runtime Activation Report

- block_id: CKA-SEC-04
- conclusion: cka_sec04_encrypted_runtime_activation_ready
- sqlcipher_provider_available: True
- sqlcipher_provider_name: sqlcipher3
- cipher_version_available: True

## Runtime activation guard

- default_runtime_encryption_active: False
- runtime_activation_default_off: True
- encrypted_runtime_flag_supported: True
- encrypted_runtime_blocks_without_key: True
- wrong_key_failure_passed: True
- test_mode_encrypted_runtime_opened: True
- test_mode_records_count_zero: True
- create_if_missing_requires_explicit_flag: True
- rollback_plan_ready: True

## Main store boundary

- main_store_migration_performed: False
- real_data_migrated: False
- existing_store_migrated: False
- real_empty_store_created_by_default: False

## Safety / privacy

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

- Case A: [PASS] Baseline SEC-03A confirmed
- Case B: [PASS] Default runtime unencrypted
- Case C: [PASS] Runtime flag without key blocks
- Case D: [PASS] Runtime wrong key fails
- Case E: [PASS] Test-mode encrypted runtime
- Case F: [PASS] create_if_missing false blocks missing store
- Case G: [PASS] Rollback plan ready
- Case H: [PASS] Main store untouched
- Case I: [PASS] Final CKA validation invocable
- Case J: [PASS] Report safety

## Next recommended block

CKA-SEC-05 Operator Runtime Launch Script, only if operator wants one-click encrypted-runtime startup
