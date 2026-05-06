# CKA-SEC-05 Encrypted Runtime Launcher Guide

This guide explains how an operator launches MedAI with the SQLCipher
encrypted runtime turned on. The default unencrypted launcher
(`Start_MedAI_UI.bat`) is **unchanged** — encrypted runtime is
opt-in.

---

## Two launchers, side by side

| Launcher | Mode | When to use |
|---|---|---|
| `Start_MedAI_UI.bat` *(default, unchanged)* | Local-only, **unencrypted** `MKBStore` | Day-to-day use, demos, no operator key needed. |
| `Start_MedAI_UI_Encrypted.bat` *(new, opt-in)* | Local-only, **SQLCipher-encrypted** `EncryptedCKAStore` | Only when the operator has an encryption key and wants encrypted persistence. |

Switching is reversible at any time: closing the encrypted Streamlit
window and double-clicking `Start_MedAI_UI.bat` restores the default
unencrypted runtime.

---

## How the encrypted launcher handles the key

- The encryption key is **prompted twice** by Python `getpass.getpass`.
- `getpass` does not echo the key to the terminal.
- The launcher refuses to continue if the two entries do not match.
- The launcher refuses an empty key.
- The key is **never** accepted on the command line — `--key=...` and
  `--encryption-key=...` exit non-zero with a safe error message.
- The key is **never** stored in `.env`, `config.json`, Markdown, log
  files, the public report, or Git.
- The key is passed only to the Streamlit **child process** environment;
  the parent shell's `os.environ` is **not** modified.
- A lost key cannot be recovered. The encrypted DB will be unreadable
  forever. Store the key in a password manager.

---

## Normal launch

### Windows

Double-click `Start_MedAI_UI_Encrypted.bat`, or from a terminal:

```bat
Start_MedAI_UI_Encrypted.bat
```

The wrapper calls:

```bat
python scripts\start_cka_encrypted_runtime_ui.py ^
    --store-path data\secure\cka_encrypted_future_store.db ^
    --port 8501
```

You will be prompted twice for the encryption key; nothing is echoed.
Streamlit then starts on `http://localhost:8501` with the encrypted
store wired in.

### Any platform

```bash
python scripts/start_cka_encrypted_runtime_ui.py \
    --store-path data/secure/cka_encrypted_future_store.db \
    --port 8501
```

---

## First-time creation of the encrypted future store

If `data/secure/cka_encrypted_future_store.db` does not yet exist,
add `--create-if-missing` once:

```bash
python scripts/start_cka_encrypted_runtime_ui.py \
    --store-path data/secure/cka_encrypted_future_store.db \
    --create-if-missing
```

The launcher will:

1. Prompt for the key twice via `getpass`.
2. Create an empty SQLCipher-encrypted DB at the configured path with
   the SEC-03A initializer (zero record rows, one non-sensitive
   `store_initialized` ledger event, six non-sensitive metadata rows).
3. Open the encrypted runtime against that fresh DB.
4. Launch Streamlit.

After the first run, drop `--create-if-missing` for subsequent runs;
the Windows wrapper does not include it by default.

---

## Validating without launching

You can validate the launcher without starting Streamlit:

```bash
# Parse + provider check; exit. Never creates a real DB.
python scripts/start_cka_encrypted_runtime_ui.py --dry-run

# Open an encrypted runtime against a synthetic temp DB; verify; exit.
# Uses CKA_SEC04_TEST_KEY env var (test-only).
export CKA_SEC04_TEST_KEY=$(python -c "import secrets; print('synth_op_'+secrets.token_hex(16))")
python scripts/start_cka_encrypted_runtime_ui.py --self-test --test-mode
unset CKA_SEC04_TEST_KEY
```

Neither path touches `data/secure/`.

---

## Rollback to the default unencrypted launcher

The encrypted runtime is **process-scoped** — closing the Streamlit
window stops it. To restart in the default mode:

1. Close the encrypted Streamlit window (`Ctrl+C` in the terminal it
   spawned, or close the browser + window).
2. (Optional) Unset `CKA_SEC04_TEST_KEY` if you used test-mode.
3. Double-click `Start_MedAI_UI.bat` (or run
   `python -m streamlit run app/main.py --server.port 8501`).
4. Confirm the Clinical Knowledge Safety tab loads and all 9 panels
   render. Confirm `runtime_encryption_active=False` in the snapshot.

Rollback is **non-destructive**: it does NOT delete the encrypted DB,
it does NOT delete the unencrypted store. See
`clinical_knowledge.security.runtime_rollback.RuntimeRollbackPlan`
for the seven-step plan.

---

## Hard rules — all flags below remain `false` under SEC-05

| Boundary | Value |
|---|---|
| `default_launcher_unchanged` | **true** *(`Start_MedAI_UI.bat` untouched)* |
| `encrypted_runtime_default_off` | **true** |
| `encrypted_launcher_contains_key` | **false** |
| `command_line_key_rejected` | **true** |
| `key_prompt_twice_required` | **true** |
| `key_mismatch_refused` | **true** |
| `empty_key_refused` | **true** |
| `encrypted_runtime_only_child_process_env` | **true** |
| `existing_data_migrated` | **false** |
| `main_store_migration_performed` | **false** |
| `real_empty_store_created_by_default` | **false** *(only if you pass `--create-if-missing`)* |
| `db_file_staged` | **false** *(neither this guide nor the validation stages any DB file)* |
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

Do not use the encrypted runtime to:

- upload real medical documents until operator policy approves it
- diagnose patients
- prescribe medication or determine doses
- replace a clinician's review
- enable real external connectors (the local-only privacy gate stays on)

These uses remain out of scope for the MVP scaffold and would require
their own separately-scoped, separately-approved roadmap track with a
fresh safety review.
