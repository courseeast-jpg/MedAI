# CKA-SEC-07 Encrypted Store Backup / Restore Operator Guide

This guide explains how an operator backs up and restores the
SQLCipher-encrypted CKA future store. The backup is a **byte-level
copy of an already-encrypted file** — the backup is itself encrypted,
the operator key is never written to disk, and no real data is migrated.

The default unencrypted launcher (`Start_MedAI_UI.bat`) is unaffected.
The encrypted launcher (`Start_MedAI_UI_Encrypted.bat`) is unaffected.

---

## What the tools do

| Script | Purpose |
|---|---|
| `scripts/cka_encrypted_store_backup.py` | Create an encrypted backup of a SQLCipher store. |
| `scripts/cka_encrypted_store_restore.py` | Restore an encrypted backup to a target path, verify the SHA-256 checksum, and confirm the operator key opens the restored DB. |

Both scripts:

- Prompt the operator key TWICE via `getpass`. Refuse on mismatch.
- Refuse `--key` / `--encryption-key` on the command line.
- Never echo, store, or log the key.
- Never write the key to a manifest, report, or log file.
- Support `--dry-run` to exercise the round-trip on synthetic temp DBs
  without touching any real path.

The backup never decrypts the source. The operator key is used only
to verify the source is openable (and to count records) before the
copy proceeds. The backup file inherits the source's at-rest
encryption.

---

## Backup workflow

### Dry-run

Always start here:

```bash
python scripts/cka_encrypted_store_backup.py --dry-run
```

The script creates a temp encrypted source DB, inserts a synthetic
record, byte-copies it to a temp backup, writes a sibling manifest,
prints a public-safe summary, and removes both temp files. No real
path is touched.

### Real backup

```bash
python scripts/cka_encrypted_store_backup.py \
    --source data/secure/cka_encrypted_future_store.db \
    --backup data/secure/backups/cka_encrypted_future_store_2026-05-06.bundle.db
```

The script will:

1. Refuse if the source is missing.
2. Refuse if the backup target exists (use `--overwrite` only after
   deliberate operator confirmation).
3. Prompt the operator key twice via `getpass`.
4. Open the source with the key + count records (correct-key probe).
5. Byte-copy the source bytes to the backup.
6. Compute SHA-256(backup), keep only the first 16 hex chars in the
   manifest (so it cannot match the B02 SECRET regex).
7. Run a defense-in-depth plaintext-absence check on the backup bytes.
8. Write the sibling manifest as
   `<backup_name>.backup-manifest.json` (no key, only safe hashes,
   record count, checksum prefix, timestamp).

The backup file and the manifest are **not staged** by the validation
script and **not committed** by SEC-07. Treat them as operator-local
artifacts — store them on the same machine as your password-manager
key entry.

---

## Restore workflow

### Dry-run

```bash
python scripts/cka_encrypted_store_restore.py --dry-run
```

Creates a synthetic source, backs it up, restores it to a fresh temp
target, verifies the round-trip, and removes all three temp files.

### Real restore

```bash
python scripts/cka_encrypted_store_restore.py \
    --backup data/secure/backups/cka_encrypted_future_store_2026-05-06.bundle.db \
    --target data/secure/cka_encrypted_future_store.db
```

The script will:

1. Refuse if the backup is missing.
2. Refuse if the manifest sibling is missing.
3. Refuse if the target exists (use `--overwrite` only after
   deliberate operator confirmation).
4. Prompt the operator key twice via `getpass`.
5. Byte-copy the backup to the target.
6. Compute SHA-256(target) and compare with the manifest's expected prefix.
7. Open the target with the operator key. **Wrong key raises** —
   there is no silent fallback to the unencrypted store.
8. Confirm the restored record count matches the manifest.

If any step fails, the script exits non-zero. The encrypted target on
disk is left in place so the operator can investigate; the restore
**never** rolls back automatically.

---

## Hard rules

| Boundary | Value |
|---|---|
| `backup_tool_ready` | true |
| `restore_tool_ready` | true |
| `dry_run_supported` | true |
| `correct_key_restore_passed` | true (verified during dry-run) |
| `wrong_key_restore_failed` | true (verified during validation) |
| `checksum_verified` | true |
| `plaintext_absence_verified` | true |
| `real_data_touched` | **false** |
| `real_store_modified` | **false** |
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

Do not use the backup / restore tools to:

- migrate real production data into the encrypted future store
- hand encrypted backups to third parties
- replace your password manager — the operator key is still your
  responsibility
- enable real connectors / external APIs / real terminology data /
  real LLM enrichment

These activities are out of scope and would require their own
separately-scoped, separately-approved roadmap track.

---

## Recovery warning

If the operator key is lost, the encrypted backup cannot be opened
even by the original creator. The byte-level copy preserves SQLCipher
encryption exactly as the source had it, so a lost key means the
backup is permanently unreadable. Store the key in a password manager.
