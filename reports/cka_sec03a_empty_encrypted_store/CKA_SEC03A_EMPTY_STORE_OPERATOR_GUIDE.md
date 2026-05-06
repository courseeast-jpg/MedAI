# CKA-SEC-03A — Empty Encrypted Future Store Operator Guide

This guide explains how an operator creates the **empty** SQLCipher-encrypted
future CKA store. It does **not** migrate any existing data, does **not**
activate the new store at runtime, and does **not** modify the production
MKBStore.

The recommended default is to **not** create the real DB at all in this
block. SEC-03A's synthetic temp-store rehearsal is what proves the
initializer works. Real-store creation requires explicit operator approval.

---

## What this block does (and does NOT do)

This block:

- Creates an empty SQLCipher-encrypted database with a future-store schema
  (`cka_future_records`, `cka_future_ledger`, `cka_future_metadata`).
- Inserts **zero** record rows.
- Inserts **one** non-sensitive ledger event marker (`store_initialized`).
- Inserts six non-sensitive metadata rows (schema version, kind, created-at,
  and three negative-assertion flags).
- Writes a sibling JSON manifest containing only safe hashes and metadata —
  never the path, never the key.

This block **never**:

- migrates the existing main CKA store
- copies real records or PHI
- switches the runtime to read from / write to the encrypted store
- accepts a key on the command line
- writes the key to a report, env file, or git
- activates external APIs
- modifies OCR / extractor / safety gates / clinical logic
- reopens the frozen HITL release

---

## Recommended path: dry-run only

Most operators should run only the dry-run mode. This creates a temp DB,
proves the initializer is healthy, and removes the temp DB after the
validation script finishes.

```bash
python scripts/init_cka_empty_encrypted_store.py --dry-run
```

This uses a synthetic in-process key, writes the temp DB to the system
temp directory, and prints a safe summary. The repo is untouched.

To re-confirm overall readiness, run the SEC-03A validation script:

```bash
python scripts/run_cka_sec03a_empty_encrypted_store_validation.py
```

Expected outcome: 10 / 10 cases pass; `real_empty_store_created: False`.

---

## Optional path: create the REAL empty future store

This path actually creates a file at:

```
data/secure/cka_encrypted_future_store.db
```

Only run this if all of the following are true:

1. You have the responsible authority's written approval.
2. The encryption key is generated outside this terminal session
   (e.g. in a password manager) and you can paste it via getpass twice.
3. You understand that **a lost key cannot be recovered**.
4. You are NOT using this store at runtime — runtime activation is
   deferred to **CKA-SEC-04**.
5. No active Streamlit / pipeline process is using any CKA DB.

Run:

```bash
python scripts/init_cka_empty_encrypted_store.py \
    --target data/secure/cka_encrypted_future_store.db \
    --approve-real-store-creation
```

The script will:

- Refuse if the target already exists (use `--overwrite` only after
  separate written approval — overwriting outside the temp dir requires
  `--approve-real-store-creation` as well).
- Create the parent directory `data/secure/` if missing.
- Create a `cka_encrypted_future_store.init.lock` file during init and
  remove it on success or failure.
- Prompt for the encryption key TWICE via getpass.
- Refuse if the two entries do not match.
- Refuse an empty or hardcoded key.
- Apply `PRAGMA key` before any schema operation.
- Create the empty schema.
- Verify correct-key open, wrong-key failure, and plaintext absence.
- Write the safe manifest sibling JSON.
- Print only the public-safe summary.

The DB file and the manifest sibling are **not staged** by default.
Keep them on the local machine. Do not commit either.

---

## Test-mode (CI / automated tests)

Test-mode is intended for automated test runs, not for production
operators.

```bash
export CKA_SEC03A_TEST_KEY=...   # synthetic key only
python scripts/init_cka_empty_encrypted_store.py \
    --target /tmp/cka_sec03a_test.db \
    --approve-real-store-creation \
    --test-mode
```

The test-mode env var is read once and never written to any report. The
target path **must** be a temp path; this guide does not authorise
test-mode against `data/secure/`.

---

## Safety summary

| Boundary | Value |
|---|---|
| empty_future_store_runtime_active | **false** |
| main_store_migration_performed | **false** |
| real_data_migrated | **false** |
| real_existing_store_migrated | **false** |
| encryption_key_logged | **false** |
| key_stored_in_repo | **false** |
| db_file_staged | **false** |
| external_api_used | **false** |
| frozen_hitl_release_reopened | **false** |

The next safe block is **CKA-SEC-04 Encrypted Store Runtime Activation**,
opened only after a separate operator approval cycle. Until then, the
production CKA scaffold continues to use the existing unencrypted
in-memory / SQLite path; SEC-03A only creates the destination, it does
not switch the runtime.
