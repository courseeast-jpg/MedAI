# CKA-SEC-02 Main Store Migration Plan Report

- block_id: CKA-SEC-02
- conclusion: cka_sec02_main_store_migration_plan_ready
- sqlcipher_provider_available: True
- sqlcipher_provider_name: sqlcipher3
- cipher_version_available: True

## Synthetic migration rehearsal

- rehearsal_passed: True
- records_copied: 3
- correct_key_read_passed: True
- wrong_key_failure_passed: True
- plaintext_absence_verified: True
- source_unchanged: True
- temp_db_files_staged: False

## Policy state

- key_management_policy_ready: True
- backup_policy_ready: True
- rollback_policy_ready: True
- operator_approval_checklist_created: True

## Main store boundary

- real_migration_approved: False
- main_store_migration_performed: False
- real_data_migrated: False
- sqlcipher_encryption_active_for_main_store: False

## Inventory (safe hashes only)

- candidate_db_count: 2
- likely_main_store_found: True
- real_main_store_touched: False
- raw_paths_written_to_public_report: False

## Safety / privacy

- external_api_used: False
- raw_phi_logged_in_public_reports: False
- private_filename_path_leaks: 0
- secret_leaks: 0
- encryption_key_logged: False
- clinical_recommendations_generated: False
- prescription_dosing_advice_generated: False
- production_ocr_changed: False
- production_extractor_changed: False
- safety_gate_changed: False
- frozen_hitl_release_reopened: False

## Cases

- Case A: [PASS] SEC-01A baseline confirmed
- Case B: [PASS] Inventory read-only
- Case C: [PASS] Key handling readiness
- Case D: [PASS] Backup and rollback readiness
- Case E: [PASS] Synthetic migration rehearsal
- Case F: [PASS] Main store untouched
- Case G: [PASS] Final CKA validation invocable
- Case H: [PASS] Report safety

## Next recommended block

CKA-SEC-03 Main Store Migration Execution, only after explicit operator approval (see CKA_SEC02_OPERATOR_MIGRATION_APPROVAL_CHECKLIST.md)
