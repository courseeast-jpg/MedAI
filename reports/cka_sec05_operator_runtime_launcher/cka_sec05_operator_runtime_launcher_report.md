# CKA-SEC-05 Operator Runtime Launcher Report

- block_id: CKA-SEC-05
- conclusion: cka_sec05_operator_runtime_launcher_ready
- sqlcipher_provider_available: True
- sqlcipher_provider_name: sqlcipher3

## Launcher state

- default_launcher_unchanged: True
- encrypted_launcher_created: True
- encrypted_launcher_contains_key: False
- python_launcher_created: True
- command_line_key_rejected: True
- key_prompt_twice_required: True
- key_mismatch_refused: True
- empty_key_refused: True
- dry_run_supported: True
- self_test_supported: True
- create_if_missing_gated: True
- encrypted_runtime_default_off: True
- encrypted_runtime_only_child_process_env: True

## Boundaries

- real_empty_store_created_by_default: False
- existing_data_migrated: False
- main_store_migration_performed: False
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

- Case A: [PASS] Baseline SEC-04 + final CKA confirmed
- Case B: [PASS] Default launcher unchanged
- Case C: [PASS] Encrypted wrapper safe
- Case D: [PASS] CLI rejects key args
- Case E: [PASS] Key mismatch refused
- Case F: [PASS] Empty key refused
- Case G: [PASS] Dry-run / self-test
- Case H: [PASS] Missing store without create-if-missing blocks
- Case I: [PASS] create-if-missing gated
- Case J: [PASS] Report safety

## Next recommended action

stop; use encrypted launcher only when operator needs encrypted runtime
