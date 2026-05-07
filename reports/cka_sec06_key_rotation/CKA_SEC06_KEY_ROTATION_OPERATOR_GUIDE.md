# CKA-SEC-06 Operator Key Rotation Guide

This guide explains how an operator rotates the SQLCipher encryption
key on an existing encrypted CKA store. SEC-06 builds on top of SEC-07
backup/restore — a verified backup is created BEFORE every rotation
attempt.

The default unencrypted launcher (`Start_MedAI_UI.bat`) is unaffected.
The encrypted launcher (`Start_MedAI_UI_Encrypted.bat`) is unaffected.
SEC-06 adds rotation tooling only — it does **not** change runtime
behaviour and does **not** rotate any real key by default.

---

## What the tool does

`scripts/cka_encrypted_store_rotate_key.py`:

1. Prompts the **old** encryption key once via `getpass`.
2. Prompts the **new** encryption key TWICE via `getpass`. Refuses on
   mismatch.
3. Refuses an empty old or empty new key.
4. Refuses if the new key equals the old key.
5. Refuses to operate on a non-temp DB path unless
   `--approve-real-rotation` is passed.
6. Verifies the old key opens the DB BEFORE rotation.
7. Creates an encrypted byte-level backup using SEC-07 tooling and
   verifies its SHA-256 prefix against the manifest.
8. Runs `PRAGMA rekey` in place to rewrite the file with the new key.
9. Verifies the new key opens the rotated DB and the record count is
   preserved.
10. Verifies the OLD key is rejected on the rotated DB (no silent
    fallback).
11. Verifies plaintext-absence in the rotated bytes.
12. Verifies the backup file still opens with the OLD key and the
    record count matches (rollback path).

If any of these steps fails, the tool exits non-zero. **No silent
recovery.** The backup file is left in place so the operator can
investigate and restore.

---

## Hard rules — keys are never written, never echoed

- The encryption key is **never** accepted on the command line.
- The encryption key is **never** echoed to the terminal.
- The encryption key is **never** written to disk, env file, log,
  manifest, public report, or stdout/stderr.
- `--key`, `--encryption-key`, `--old-key`, `--new-key` all exit
  non-zero with `command_line_key_not_accepted`.
- Synthetic test-mode is allowed only when both
  `CKA_SEC06_OLD_TEST_KEY` and `CKA_SEC06_NEW_TEST_KEY` env vars are
  present AND `--test-mode` is passed. Test-mode env-var values are
  never written to a public report.

---

## Recommended workflow

### 1. Dry-run

Always run a dry-run rehearsal first:

```bash
python scripts/cka_encrypted_store_rotate_key.py --dry-run
```

This creates a synthetic temp encrypted DB with 3 non-PHI records,
exercises the entire backup → rotate → verify → rollback chain in
temp paths only, and removes all temp files. Expected output ends with
`rotation_performed: True` and a populated set of safe summary fields.

### 2. Real rotation (only after explicit approval)

When the responsible authority approves, run:

```bash
python scripts/cka_encrypted_store_rotate_key.py \
    --db-path data/secure/cka_encrypted_future_store.db \
    --backup-path data/secure/backups/cka_encrypted_future_store_pre_rekey_<UTC>.bundle.db \
    --approve-real-rotation
```

The tool will:

- Refuse if the encrypted Streamlit runtime is currently running on the
  target DB. (Operator must shut down `Start_MedAI_UI_Encrypted.bat`
  before rotating.)
- Prompt for the old key once and the new key twice via `getpass`.
- Create the verified backup at `--backup-path`.
- Perform the rotation.
- Verify the new key works, the old key fails, plaintext is absent,
  and the backup still opens with the old key.

After rotation, restart the encrypted Streamlit runtime with the NEW
key. The OLD key will no longer open the rotated DB. The backup at
`--backup-path` retains the OLD encryption — keep it as a recovery
artifact.

---

## Rollback plan

1. Backup MUST be created and SHA-256 verified BEFORE any rotation runs.
2. If the new key cannot open the rotated DB, STOP using the rotated DB.
3. Restore the verified encrypted backup using
   `cka_encrypted_store_restore.py`.
4. Confirm the backup opens with the OLD key and the record count
   matches.
5. Do NOT delete the backup until the operator has verified the new
   key opens the rotated DB AND the app starts correctly.
6. If both old and new keys fail, ESCALATE to the responsible
   authority. Do not attempt destructive repair.
7. Encryption keys are never stored in Git, never written to reports,
   never logged. **A lost key means the data is unrecoverable.**

---

## Hard-rule flags (binding under SEC-06)

| Boundary | Value |
|---|---|
| `key_rotation_tool_ready` | true |
| `synthetic_rotation_rehearsal_passed` | true (verified during validation) |
| `backup_before_rotation_required` | **true** |
| `backup_checksum_verified` | true |
| `new_key_open_after_rotation_passed` | true |
| `old_key_rejected_after_rotation` | true |
| `record_count_preserved` | true |
| `rollback_restore_verified` | true |
| `plaintext_absence_verified` | true |
| `command_line_keys_rejected` | **true** |
| `empty_key_refused` | **true** |
| `key_mismatch_refused` | **true** |
| `same_old_new_key_refused` | **true** |
| `real_rotation_blocked_by_default` | **true** |
| `real_rotation_performed` | **false** |
| `real_store_touched` | **false** |
| `real_data_touched` | **false** |
| `db_file_staged` | **false** |
| `key_stored_in_repo` | **false** |
| `encryption_key_logged` | **false** |
| `external_api_used` | **false** |
| `clinical_recommendations_generated` | **false** |
| `prescription_dosing_advice_generated` | **false** |
| `production_ocr_changed` | **false** |
| `production_extractor_changed` | **false** |
| `safety_gate_changed` | **false** |
| `frozen_hitl_release_reopened` | **false** |

---

## What this scaffold is NOT for

Do not use the rotation tool to:

- migrate real production data into a freshly-keyed DB
- rotate keys held by third parties
- replace your password manager — both old and new keys remain the
  operator's responsibility
- enable real connectors / external APIs / real terminology data /
  real LLM enrichment

Any of those activities require their own separately-scoped,
separately-approved roadmap track.
