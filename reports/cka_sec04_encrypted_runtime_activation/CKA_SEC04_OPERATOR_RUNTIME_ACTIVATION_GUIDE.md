# CKA-SEC-04 — Operator Runtime Activation Guide

This guide explains how an operator turns the SQLCipher-encrypted CKA
runtime **on** (and **off**), if and when it is approved. The scaffold's
default behaviour is unchanged: the encrypted runtime is **OFF** and the
existing `MKBStore` is the active store.

CKA-SEC-04 only adds a *guard* and a *factory*. It does **not**:

- migrate any existing data
- copy real records
- read real patient data
- create encrypted clinical recommendations
- activate real connectors / external APIs / UMLS / SNOMED / LLM
- modify OCR / extractor / safety-gate behaviour
- delete or overwrite any DB file

---

## Default state

With no environment flags set:

```
runtime_encryption_active      = False
default_runtime_encryption     = OFF
existing MKBStore behaviour    = unchanged
```

The MedAI Streamlit app behaves exactly as it did at the operator-ready
seal `medai-cka-operator-ready-2026-05-06`. No operator action is
required if the encrypted runtime is not desired.

---

## Activation environment flags

To turn the encrypted runtime on, set the following env vars before
launching MedAI:

| Variable | Required? | Purpose |
|---|---|---|
| `MEDAI_CKA_ENCRYPTED_STORE_ENABLED=1` | yes | Master switch. Without this, encryption is OFF. |
| `MEDAI_CKA_ENCRYPTED_STORE_PATH=...` | yes | Path to the encrypted DB file. Use the SEC-03A path if available. |
| `MEDAI_CKA_ENCRYPTION_KEY=...` | yes | Operator-supplied key. **Must satisfy** the SEC-02 key policy (>= 12 chars, no hardcoded marker). |
| `MEDAI_CKA_ENCRYPTED_STORE_CREATE_IF_MISSING=1` | optional | Allow the factory to create an empty encrypted store at the path if it does not exist. Default OFF. |

Important:

- The encryption key is **never** logged, **never** written to a public
  report, **never** committed to Git.
- The factory **refuses to silently fall back** to the unencrypted
  `MKBStore` if you requested the encrypted runtime and a wrong key,
  missing key, or missing path is supplied. You will see an explicit
  `RuntimeFactoryError` instead.
- If the encrypted store does **not** exist at the configured path and
  `MEDAI_CKA_ENCRYPTED_STORE_CREATE_IF_MISSING` is not set, the factory
  will refuse to create it. Use the SEC-03A initializer (with
  `--approve-real-store-creation`) to create it once, separately, with
  the operator approval flow.

---

## Recommended order of operations

1. Operator records the encryption key in a password manager.
2. Operator runs the SEC-03A initializer once (interactive, getpass
   twice) to create the empty encrypted store at
   `data/secure/cka_encrypted_future_store.db`. SEC-04 does **not**
   need to do this; it can re-use whatever SEC-03A produced.
3. Operator sets the four env vars (`*_ENABLED`, `*_PATH`,
   `*_ENCRYPTION_KEY`, optionally `*_CREATE_IF_MISSING`).
4. Operator launches MedAI with `Start_MedAI_UI.bat` (or the platform
   equivalent). The runtime factory now returns the encrypted store.
5. Operator confirms the Clinical Knowledge Safety tab still loads.

If anything is wrong with the configuration, the runtime factory
**raises** before the app finishes starting; the operator will see the
error in the terminal that launched the app.

---

## Rollback

Rollback is **non-destructive**: it re-routes the runtime back to the
unencrypted `MKBStore` and leaves both the encrypted DB and the
unencrypted store untouched.

```
1. Unset MEDAI_CKA_ENCRYPTED_STORE_ENABLED in the operator's environment.
2. Optionally unset MEDAI_CKA_ENCRYPTION_KEY and MEDAI_CKA_ENCRYPTED_STORE_PATH.
3. Restart MedAI.
4. Confirm the Clinical Knowledge Safety tab loads and all 9 panels render.
5. Confirm runtime_encryption_active=False and external_api_used=False.
6. Do NOT delete the encrypted store file.
7. Do NOT delete the existing unencrypted store.
8. Document the rollback in the operator-review log.
```

The rollback steps above are also returned programmatically by
`get_runtime_rollback_plan()`.

---

## Safety boundaries (binding)

| Boundary | Value |
|---|---|
| `default_runtime_encryption_active` | **false** |
| `runtime_activation_default_off` | **true** |
| `encrypted_runtime_blocks_without_key` | **true** |
| `wrong_key_failure_passed` | **true** |
| `create_if_missing_requires_explicit_flag` | **true** |
| `main_store_migration_performed` | **false** |
| `real_data_migrated` | **false** |
| `existing_store_migrated` | **false** |
| `real_empty_store_created_by_default` | **false** |
| `db_file_staged` | **false** |
| `key_stored_in_repo` | **false** |
| `encryption_key_logged` | **false** |
| `external_api_used` | **false** |
| `frozen_hitl_release_reopened` | **false** |
| `production_ocr_changed` | **false** |
| `production_extractor_changed` | **false** |
| `safety_gate_changed` | **false** |
| `clinical_recommendations_generated` | **false** |
| `prescription_dosing_advice_generated` | **false** |

The next safe block is **CKA-SEC-05 Operator Runtime Launch Script**,
opened only if the operator wants a one-click encrypted-runtime
startup. Until then, the env-var-driven flow described above is the
only authorised path.
