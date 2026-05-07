# MEDAI FINAL POST-SEC-06 PARKING SNAPSHOT — ENCRYPTED RUNTIME + BACKUP + KEY ROTATION TOOLING READY

This is the binding hand-off document for the MedAI repository at the
moment it was re-parked after the CKA security-hardening track stopped
at SEC-06 (operator key rotation plan + synthetic rehearsal). It
supersedes MEDAI-PARK-03 in the "what's new" sense — the operator-ready
seal, every prior security-hardened tag, and the post-SEC-05 parked
tag remain valid and unmoved.

---

## Identity

- **Branch:** `clinical-knowledge-architecture`
- **Current HEAD (pre-parking-commit):** `d5b692d` — CKA-SEC-06
- **Final post-SEC-06 parking commit:** added by `MEDAI-PARK-04` (this snapshot's commit)

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
| CKA-SEC-07 | `c73e846` | Encrypted store backup / restore tooling |
| CKA-SEC-06 | `d5b692d` | Operator key rotation plan + synthetic rehearsal |

- Pre-existing tags (all unmoved):
  - `medai-cka-security-hardened-2026-05-06` → `0026378`
  - `medai-cka-final-parked-2026-05-06` → `34a80cc`
  - `medai-cka-sec05-encrypted-launcher-ready-2026-05-06` → `3803bb9`
  - `medai-cka-final-parked-post-sec05-2026-05-06` → `b0c65c5`
- **New tags created in this block:**
  - `medai-cka-sec06-key-rotation-ready-2026-05-06` → `d5b692d`
  - `medai-cka-final-parked-post-sec06-2026-05-06` → MEDAI-PARK-04 commit

---

## Current launcher / runtime state

| Launcher | Mode | Status |
|---|---|---|
| `Start_MedAI_UI.bat` | Local-only, **unencrypted** `MKBStore` | **default**, **unchanged** by SEC-04/05/06/07 |
| `Start_MedAI_UI_Encrypted.bat` | Local-only, **SQLCipher-encrypted** `EncryptedCKAStore` | **opt-in only**, unchanged by SEC-06/07 |

| Boundary | Value |
|---|---|
| `encrypted_runtime_default_off` | **true** |
| `existing_data_migrated` | **false** |
| `main_store_migration_performed` | **false** |
| `real_empty_store_created_by_default` | **false** |
| `real_key_rotation_performed` | **false** |
| `real_store_touched` | **false** |
| `real_data_touched` | **false** |
| `external_api_used` | **false** |
| `real_connectors_active` | **false** |
| `clinical_logic_changed` | **false** |
| `production_ocr_changed` | **false** |
| `production_extractor_changed` | **false** |
| `safety_gate_changed` | **false** |
| `frozen_hitl_release_reopened` | **false** |
| `db_file_staged` | **false** |
| `key_stored_in_repo` | **false** |
| `encryption_key_logged` | **false** |

---

## Validations (this snapshot)

- **CKA-SEC-06 key rotation:** PASS — 10 / 10 cases.
- **CKA-SEC-07 backup / restore:** PASS — 12 / 12 cases.
- **CKA-SEC-05 launcher:** PASS — 10 / 10 cases.
- **CKA-B11 final MVP release:** PASS — 12 / 12 cases, 26 / 26 preflight, **693 tests passed**.
- **CKA-B02 public report privacy checker** on this snapshot: PASS.

---

## Stop decision

**Stop after SEC-06.** Do **not** perform a real key rotation against
`data/secure/cka_encrypted_future_store.db` (or any other real
encrypted store) without explicit operator approval recorded through
the responsible-authority channel.

The synthetic rotation rehearsal is verified end-to-end:

- backup created and SHA-256 verified before rotation
- `PRAGMA rekey` runs in place
- new key opens the rotated DB
- old key is rejected on the rotated DB
- record count preserved
- plaintext absence in rotated bytes
- rollback restore from backup with the OLD key still works

But:

- no real DB exists in the repo
- no real rotation has been performed
- the encrypted runtime itself remains opt-in
- the existing unencrypted store is intact
- no real data has been migrated, copied, or read
- no real external connectors / APIs / terminology / LLM are active

---

## Future possible tracks

Listed lowest-risk to highest-risk; each requires a separately scoped
roadmap, separate operator approval, and separate safety review.

1. **Real terminology integration (UMLS / SNOMED / RxNorm)**
   — license-checked subset import, validated coding tables, periodic
   refresh, version tracking.
2. **Russian / multilingual support**
   — extend extractor + sanitizer + coder for Cyrillic and other
   scripts, language-tagged records, locale-aware DDI checks.
3. **Local LLM activation**
   — enable local-only LLM enrichment behind `ENABLE_LOCAL_LLM`, with
   sandboxed prompts, output sanitation, and hypothesis-only write
   boundary preserved.
4. **Real connector activation (DxGPT / SAGE / PatientNotes / others)**
   — highest-risk; mandatory per-connector privacy review, per-connector
   enable flag, operator approval per outbound call.
5. **Real encrypted-store operational use**
   — only after explicit policy approval, after the operator has
   followed `CKA_SEC03A_OPERATOR_GUIDE.md`, `CKA_SEC05_LAUNCHER_GUIDE.md`,
   `CKA_SEC07_BACKUP_RESTORE_OPERATOR_GUIDE.md`, and
   `CKA_SEC06_KEY_ROTATION_OPERATOR_GUIDE.md`. Until then this remains a
   tooling-ready scaffold, not an active encrypted production store.

The default recommendation, absent further approval, is to leave the
scaffold parked at this snapshot and not begin any of the above.
