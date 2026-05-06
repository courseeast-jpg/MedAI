# MEDAI FINAL POST-SEC-05 PARKING SNAPSHOT — ENCRYPTED LAUNCHER READY

This is the binding hand-off document for the MedAI repository at the
moment it was re-parked after the CKA security-hardening track stopped
at SEC-05 (encrypted runtime operator launcher). It supersedes the
MEDAI-PARK-01 snapshot only in the "what's new" sense — the earlier
operator-ready and security-hardened tags remain valid and unmoved.

---

## Identity

- **Branch:** `clinical-knowledge-architecture`
- **Current HEAD (pre-parking-commit):** `3803bb9` — CKA-SEC-05
- **Final post-SEC-05 parking commit:** added by `MEDAI-PARK-03` (this snapshot's commit)

---

## OCR / Layout HITL frozen release

- **Phase:** `Phase78`
- **Final HITL commit:** `3c0c869`
- **Status:** `FROZEN_HITL_RELEASE`
- The OCR / extractor / safety-gate logic from this release is **not**
  modified by any CKA or SEC block. Do **not** reopen unless explicitly
  approved by the responsible authority.

---

## CKA MVP scaffold (sealed, operator-ready)

| Block | Commit | Title |
|---|---|---|
| CKA-B11 | `07860eb` | Final clinical knowledge MVP release package |
| CKA-OPR-01 | `111843a` | Operator review polish and documentation drift cleanup |
| CKA-OPS-02 | `1cfbcb1` | Release seal and operator launch verification |

- **Operator-ready tag:** `medai-cka-operator-ready-2026-05-06` → `1cfbcb1` (unmoved)

---

## Security hardening

| Block | Commit | Title |
|---|---|---|
| CKA-SEC-01 | `61baa21` | SQLCipher encryption readiness and encrypted store adapter |
| CKA-SEC-01A | `c691f70` | SQLCipher provider enabled and synthetic encryption verified |
| CKA-SEC-02 | `4af02cd` | Main store migration plan and synthetic rehearsal |
| CKA-SEC-03A | `0026378` | Encrypted empty future store initializer |
| CKA-SEC-04 | `117e9d4` | Encrypted runtime activation guard |
| CKA-SEC-05 | `3803bb9` | Encrypted runtime operator launcher |

- Security-hardened tag (unmoved): `medai-cka-security-hardened-2026-05-06` → `0026378`
- First parked tag (unmoved): `medai-cka-final-parked-2026-05-06` → `34a80cc`
- New tags created in this block:
  - `medai-cka-sec05-encrypted-launcher-ready-2026-05-06` → `3803bb9`
  - `medai-cka-final-parked-post-sec05-2026-05-06` → MEDAI-PARK-03 commit

---

## Launchers

| Launcher | Mode | Status |
|---|---|---|
| `Start_MedAI_UI.bat` | Local-only, **unencrypted** `MKBStore` | **default**, **unchanged** by SEC-04/05 |
| `Start_MedAI_UI_Encrypted.bat` | Local-only, **SQLCipher-encrypted** `EncryptedCKAStore` | **new in SEC-05**, **opt-in only** |

The encrypted launcher prompts the operator key twice via `getpass`
and never stores or logs it; the key is passed only to the Streamlit
child process's environment. The Python launcher
(`scripts/start_cka_encrypted_runtime_ui.py`) refuses any
`--key` / `--encryption-key` command-line argument with a non-zero
exit.

---

## Current runtime state

| Boundary | Value |
|---|---|
| `default_runtime_encryption_active` | **false** |
| `encrypted_runtime_default_off` | **true** |
| `encrypted_runtime_opt_in_only` | **true** |
| `existing_data_migrated` | **false** |
| `main_store_migration_performed` | **false** |
| `real_empty_store_created_by_default` | **false** |
| `external_api_used` | **false** |
| `real_connectors_active` | **false** |
| `clinical_logic_changed` | **false** |
| `production_ocr_changed` | **false** |
| `production_extractor_changed` | **false** |
| `safety_gate_changed` | **false** |
| `frozen_hitl_release_reopened` | **false** |
| `command_line_key_rejected` | **true** |
| `key_stored_in_repo` | **false** |
| `encryption_key_logged` | **false** |
| `db_file_staged` | **false** |

---

## Validations

- **CKA-SEC-05 launcher validation:** PASS — 10 / 10 cases
- **CKA-SEC-04 runtime activation validation:** PASS — 10 / 10 cases
- **CKA-B11 final MVP release validation:** PASS — 12 / 12 cases, 26 / 26 preflight, **693 tests passed**
- **CKA-B02 public report privacy checker** on this snapshot: PASS

---

## Stop decision

**Stop after SEC-05.** Do not proceed to **CKA-SEC-06 Operator Key
Rotation Plan** or any other hardening / activation block without
explicit operator approval recorded through the responsible-authority
channel.

The encrypted runtime launcher is in place, but:

- the encrypted runtime is OFF by default
- no production code path uses it unless the operator explicitly runs
  `Start_MedAI_UI_Encrypted.bat` or the Python launcher
- the existing unencrypted store is intact
- no real database file has been created in the repo
- no real data has been migrated, copied, or read
- no real external connectors / APIs / terminology / LLM are active
- the frozen HITL release is unmodified

---

## Future possible tracks

Listed lowest-risk to highest-risk; each requires a separately scoped
roadmap, separate operator approval, and separate safety review.

1. **CKA-SEC-06 Operator Key Rotation Plan**
   — define the operator workflow for rotating the SQLCipher key
   without re-creating the DB. Synthetic rehearsal only; no real data
   re-keying.
2. **CKA-SEC-07 Encrypted Store Backup / Restore Tooling**
   — guarded `--export` and `--restore` operator paths for the
   encrypted future store, including verified-checksum backups.
3. **Real terminology integration (UMLS / SNOMED / RxNorm)**
   — license-checked subset import, validated coding tables, periodic
   refresh, version tracking.
4. **Russian / multilingual support**
   — extend extractor + sanitizer + coder for Cyrillic and other
   scripts, language-tagged records, locale-aware DDI checks.
5. **Real connector activation (DxGPT / SAGE / PatientNotes / others)**
   — highest-risk track; mandatory per-connector privacy review,
   per-connector enable flag, operator approval per outbound call.

The default recommendation, absent further approval, is to leave the
scaffold parked at this snapshot and not begin any of the above.
