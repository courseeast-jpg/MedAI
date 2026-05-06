# CKA Limitations and Safety Statement

This document is the binding safety scope statement for the Clinical
Knowledge Architecture (CKA) MVP scaffold delivered through CKA-B01
through CKA-B10.

Read this before assuming any clinical capability.

---

## Scope statement

The CKA scaffold is **technical safety scaffolding**. It is:

- **NOT production-autonomous.**
- **NOT a medical device.**
- **NOT clinical diagnosis software.**
- **NOT prescribing software.**
- **NOT a substitute for licensed medical professionals.**

No part of the CKA scaffold is approved, certified, or validated for
patient care. It does not issue medical advice. It does not diagnose.
It does not prescribe medications. It does not determine dosing.

---

## External integrations

- **No real external connectors are active.**
- All connector implementations are **synthetic, local-only stubs**
  that return deterministic fixture data.
- `dxgpt_stub`, `sage_epilepsy_stub`, `patientnotes_ddi_stub`, and
  `generic_stub` do not call any external API or service.
- `ConnectorRegistry` enforces `allow_external=False` and
  `synthetic_only=True` at registration time.
- `CKAConfig.EXTERNAL_APIS_ENABLED` is `False` by default and the
  scaffold raises `ValueError` if it is set to `True`.

---

## Terminology and data sources

- **No real UMLS terminology is active.** No UMLS API key is used.
- **No real SNOMED CT data is downloaded or shipped.**
- **No scispaCy entity linker is required or invoked.**
- The Medical Coding layer uses a **local synthetic mapping table**.
- Unknown entities remain unmapped — the coder never invents codes.

---

## DDI / medication safety

- **No real PatientNotes DDI API is active.**
- DDI checks are performed by a **local synthetic stub** that returns
  fixture interactions for test purposes only.
- The scaffold does not generate medication recommendations.
- The scaffold does not generate prescription dosing advice.
- The scaffold does not issue medication orders.
- DDI status on a quarantined record is **never cleared** by the
  scaffold.

---

## LLM / enrichment

- **No real LLM API is active** for enrichment.
- `CKAConfig.ENABLE_LOCAL_LLM` is `False` by default.
- All AI-derived facts remain `hypothesis` tier.
- `CKAConfig.ENRICH_PROMOTE` is `False` by default — auto-promotion
  to `active` tier is blocked.

---

## Privacy

- The privacy boundary (CKA-B02) sanitizes PHI/PII/secret patterns
  before any outbound payload is constructed.
- `ALWAYS_BLOCK_CATEGORIES` includes `SECRET`.
- Public reports are validated by the report privacy checker before
  they are written.
- No raw filenames, paths, or replacement maps appear in any public
  CKA report.
- No `*_PRIVATE.json` files are written into the public CKA report
  directories.

---

## What the scaffold outputs are

All outputs of the CKA scaffold are **technical safety scaffolding** —
ledger events, hypothesis-tier records, quarantine markers, consensus
statuses, coding suggestions, public reports, and operator UI panels.

These outputs are **not medical interpretations**. Manual clinical
verification by a qualified medical professional is required for any
real medical interpretation, medication decision, or diagnostic claim.

---

## Frozen HITL release boundary

The CKA scaffold did **not** modify the frozen HITL OCR/Layout release.
The release status remains `FROZEN_HITL_RELEASE` and the following
flags remain at their committed values:

- `production_ocr_should_change_yet`: False
- `production_extractor_should_change_yet`: False
- `safety_gates_should_change_yet`: False
- `frozen_hitl_release_reopened`: False

CKA is delivered as a parallel scaffold on the
`clinical-knowledge-architecture` branch and does not touch the frozen
release artifacts.
