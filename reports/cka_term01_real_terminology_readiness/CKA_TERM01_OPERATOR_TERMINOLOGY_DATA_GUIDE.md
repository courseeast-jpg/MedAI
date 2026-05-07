# CKA-TERM-01 Operator Terminology Data Guide

This guide explains how an operator supplies licensed terminology data
(UMLS, SNOMED CT, RxNorm, LOINC) to the MedAI CKA scaffold for **local
inventory and lookup only**. TERM-01 does NOT download any data, does
NOT call any vendor API, and does NOT bypass licensing.

The default unencrypted launcher (`Start_MedAI_UI.bat`), the encrypted
launcher (`Start_MedAI_UI_Encrypted.bat`), and every CKA-B07 medical
coding default behaviour are **unchanged** by this block.

---

## What this block adds

| Layer | Purpose |
|---|---|
| `clinical_knowledge.terminology.file_inventory` | Read-only inventory of `terminology_data/<system>/` directories. Never opens file content. |
| `clinical_knowledge.terminology.license_gate` | Refuses real-import unless the operator has acknowledged their license. |
| `clinical_knowledge.terminology.parsers` | Streaming parsers for synthetic / small UMLS, SNOMED, RxNorm, LOINC fixtures. |
| `clinical_knowledge.terminology.local_store` | Small SQLite terminology index (in-memory by default; on-disk path `data/terminology/` is **gitignored**). |
| `clinical_knowledge.terminology.lookup_service` | Deterministic lookup. Exact / synonym / ambiguous / unmapped. **Never invents a code.** |
| `clinical_knowledge.terminology.integration` | Opt-in B07 integration helper. Default B07 behaviour is unchanged. |

---

## Hard rules

- **No network**, no vendor API, no automated download.
- **No license bypass.** Real licensed-import is gated by the operator's
  acknowledgment file (or test-mode env var for synthetic data).
- **No real terminology data is committed** to this repository. The
  `terminology_data/`, `data/terminology/`, and `**/LICENSE_ACK_PRIVATE*`
  paths are in `.gitignore`.
- **No code hallucination.** Unknown terms return UNMAPPED with empty
  matches.
- **Coding does NOT promote hypothesis** facts. Coding does **NOT**
  clear DDI status.
- The terminology index is **metadata / coding only**. It does not
  modify the MKB store.
- No clinical recommendations, no prescription dosing, no diagnosis
  text is produced by this block.

---

## Directory layout

The operator places files locally (NEVER committed) at:

```
terminology_data/
  umls/
    MRCONSO.RRF
    MRSTY.RRF
    [optional MRREL.RRF]
  snomed_ct/
    sct2_Concept_*.txt
    sct2_Description_*.txt
    [optional sct2_Relationship_*.txt]
  rxnorm/
    RXNCONSO.RRF
    [optional RXNREL.RRF]
  loinc/
    Loinc.csv
  LICENSE_ACK_PRIVATE.json    <-- gitignored, operator-supplied
```

The `LICENSE_ACK_PRIVATE.json` file is operator-supplied and must NOT
be committed. It has a minimal schema:

```json
{
  "operator_acknowledged": true,
  "acknowledged_systems": ["umls", "snomed_ct", "rxnorm", "loinc"]
}
```

The license gate reads only those two fields. License text, operator
identity, and any other content in this file are never copied into a
public report.

---

## Test-mode acknowledgment (synthetic data only)

For automated tests and the validation script, the env var
`CKA_TERM01_TEST_LICENSE_ACK=1` is honoured **only** when the caller
explicitly passes `test_mode=True`. The test-mode path never touches
real licensed data; it is used to exercise synthetic fixtures end-to-end.

---

## Workflows

### Inventory only (default; no license action)

```bash
python scripts/cka_terminology_inventory.py
```

Prints a public-safe summary listing each system's status as one of:

- `missing` — no files in `terminology_data/<system>/`
- `present_unverified` — files present but no canonical filename match
- `license_required` — files match canonical patterns but no operator
  acknowledgment yet
- `import_ready` — files match AND operator has acknowledged the license

### Synthetic / test import

Used by the validation script and automated tests. Synthetic fixtures
are in-memory only; no on-disk state changes:

```bash
python scripts/run_cka_term01_real_terminology_readiness_validation.py
```

### Real local import (operator-controlled, NOT executed by TERM-01)

Real-licensed-import is the responsibility of a follow-on TERM-02
block, opened only after explicit operator approval. TERM-01 produces
the inventory + readiness report; it does not pull large licensed
files into a real on-disk store.

---

## What this block is NOT for

- Activating real DxGPT / SAGE / PatientNotes / LLM connectors — see SEC-04..07.
- Committing licensed terminology data into the repo — that would
  violate the licenses.
- Promoting AI-derived facts to the MKB active tier — that boundary is
  preserved by CKA-B06 / B07 / B08 and is not relaxed here.
- Issuing clinical advice or dosing — not a function of this block.

The next safe block is **CKA-TERM-02 controlled local import**, opened
only when the operator has supplied licensed local files AND placed an
acknowledgment file at `terminology_data/LICENSE_ACK_PRIVATE.json`.
