# CKA-SEC-02 Operator Migration Approval Checklist

This is the binding pre-execution checklist for **CKA-SEC-03 Main Store
Migration Execution**. CKA-SEC-02 only *generates* this checklist —
SEC-02 itself does **not** migrate the main store and does **not**
require any operator approval.

The operator must explicitly tick all six items below before any real
migration of the main MKB store begins. If any item cannot be ticked
honestly, **stop**.

---

## Identity

- **Plan block:** CKA-SEC-02 Main Store Migration Plan (current)
- **Execution block (future):** CKA-SEC-03 Main Store Migration Execution
- **Branch:** `clinical-knowledge-architecture`
- **Tag (sealed MVP):** `medai-cka-operator-ready-2026-05-06`
- **SQLCipher provider:** must remain available at the moment SEC-03 runs

---

## Required confirmations

The operator confirms:

- [ ] **A verified backup of the main CKA MKB store exists.** The
      backup is timestamped, its SHA-256 checksum is recorded, and a
      restore-test on a copy has succeeded.

- [ ] **The encryption key is stored in a password manager.** The key
      is not in code, not in environment variables, not in chat
      history, not in screenshots, and not on shared filesystems. The
      operator understands that **a lost key cannot be recovered** and
      means the encrypted DB is unreadable forever.

- [ ] **The rollback plan has been read and understood.** The operator
      can execute the seven rollback steps listed in
      `clinical_knowledge/security/rollback_plan.py` from memory or
      from the printed plan.

- [ ] **The real migration is approved by the responsible authority.**
      Approval is recorded outside this checklist (operator-review log,
      change-management ticket, or equivalent).

- [ ] **No active Streamlit / pipeline process is using the
      database.** All MedAI processes touching the MKB store are
      stopped. The migration lock file is created and held by SEC-03
      for the duration of the migration.

- [ ] **This approval is for SEC-03 execution, NOT for SEC-02
      planning.** SEC-02 never asks for operator approval; if any
      SEC-02 path appears to require this checklist, that is a bug —
      stop and investigate.

---

## What this checklist does NOT authorise

This checklist authorises only the **encryption migration of the main
CKA store** under SEC-03. It does **not** authorise:

- activation of real external connectors
- activation of real UMLS / SNOMED / RxNorm data
- activation of real DxGPT / SAGE / PatientNotes / LLM APIs
- modification of OCR, extractor, or safety-gate logic
- modification of clinical decision logic, DDI logic, Truth Resolution
  logic, hypothesis-promotion behaviour, or consensus logic
- reopening of the frozen HITL release

Any of those require their own separately-scoped approval track.

---

## Recovery warning

If the encryption key is lost after a real migration, the encrypted DB
is **permanently unreadable**. The only path back to a working state
is to restore the timestamped backup taken before SEC-03 ran. If the
backup was deleted, **the data is gone**. SEC-03 must therefore refuse
to run unless every item above has been ticked.
