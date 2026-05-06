# CKA Operator Guide — MedAI Clinical Knowledge Architecture (MVP Scaffold)

This guide explains how an operator interacts with the Clinical Knowledge
Architecture (CKA) MVP scaffold. The CKA is **technical safety scaffolding**;
it is not a clinical decision tool, not a medical device, and does not issue
medical advice.

---

## 1. What the CKA scaffold does

The CKA scaffold provides a layered, auditable structure for handling
clinical-knowledge-shaped facts inside MedAI. It covers:

| Block | Role |
|---|---|
| CKA-B01 | MKB foundation — record store + immutable ledger |
| CKA-B02 | Privacy boundary — sanitizer, outbound audit, public-report check |
| CKA-B03 | Decision Engine — Safe Mode + response scoring |
| CKA-B04 | Truth Resolution — quarantine engine for contradictions |
| CKA-B05 | Medication Safety — dual-layer DDI gate (synthetic) |
| CKA-B06 | Controlled Enrichment — hypothesis tier only |
| CKA-B07 | Medical Coding — synthetic terminology mapping |
| CKA-B08 | Multi-Connector Consensus — synthetic stub connectors |
| CKA-B09 | Operator UI — Clinical Knowledge Safety panels (Streamlit) |
| CKA-B10 | Preflight + Scaffold — system readiness checker |

All connectors are local synthetic stubs. No real external APIs are called.

---

## 2. How to run preflight and final release validation

Preflight (B01-B09 module + invariant check):

```bash
python -m clinical_knowledge.preflight
```

Final MVP release validation (12 cases, includes preflight, scaffold
invariants, B01-B10 test suite, doc presence, privacy check):

```bash
python scripts/run_cka_final_mvp_release_validation.py
```

Or programmatically:

```python
from clinical_knowledge.preflight import run_cka_preflight
report = run_cka_preflight()
print(report.passed)            # True if all checks pass
print(report.safe_public_summary())
```

A passing preflight confirms:

- All B01-B09 modules import.
- Privacy SECRET category is in the always-block set.
- Connector registry rejects `allow_external=True`.
- Enrichment rejects `allow_active_write=True`.
- Operator UI snapshot loader does not read private files.
- HITL release freeze document is present.
- `EXTERNAL_APIS_ENABLED` is `False`.

---

## 3. How to view the Clinical Knowledge Safety UI tab

Launch the operator app.

**Windows (recommended):**

```bat
Start_MedAI_UI.bat
```

This launcher sets `MEDAI_LOCAL_ONLY=1`, `MEDAI_ALLOW_EXTERNAL_API=0`,
`MEDAI_REQUIRE_PII_SCRUB=1`, `MEDAI_PRIVACY_AUDIT=1`, then starts
Streamlit on `http://localhost:8501` and opens the browser.

**Direct invocation (any platform):**

```bash
python -m streamlit run app/main.py --server.port 8501
```

Switch to the **Clinical Knowledge Safety** tab. It loads only public
CKA reports for **B01-B10** (`reports/cka_block01_*` through
`reports/cka_block10_*`) and displays nine read-only safety panels:

1. MKB Status
2. Decision Engine
3. Privacy
4. Truth Resolution
5. Medication Safety
6. Enrichment / Hypothesis
7. Medical Coding
8. Multi-Connector Consensus
9. Release Readiness

The panel layer never loads private mapping files or raw connector
responses. If a public report is missing, the panel reports "report
unavailable" rather than guessing.

---

## 4. What "Safe Mode" means

Safe Mode is the default operating posture of the Decision Engine. Under
Safe Mode:

- Low-confidence model responses are discarded, not promoted.
- Active writes are blocked.
- Refusal paths are preferred over speculation.
- All AI-derived facts remain in `hypothesis` tier.

Safe Mode is on by default and cannot be disabled by the scaffold itself.

---

## 5. What "hypothesis tier" means

Every fact in the Multi-Knowledge Base is assigned a `KnowledgeTier`:

- `active` — operator-validated, retrievable as a real medical fact.
- `hypothesis` — AI-derived or unverified; never used as ground truth.
- `quarantined` — flagged for contradiction or safety conflict.
- `superseded` — replaced by a newer validated record.

The CKA scaffold writes only `hypothesis` tier facts from any AI/connector
input. Promotion to `active` requires an explicit operator review path
that is **not** auto-triggered by the scaffold.

---

## 6. What "quarantine" means

When Truth Resolution detects a contradiction (for example, two
connectors disagreeing on a medication dose), the affected records are
moved to the `quarantined` tier. Quarantined records:

- Are **never** retrievable as active facts.
- Set `requires_review = True`.
- Generate an immutable ledger event (`QUARANTINE`).
- Are **not** auto-resolved by the scaffold.

DDI status on a quarantined medication record is **never cleared** by
the scaffold.

---

## 7. What "DDI blocked" / "DDI pending" means

The Medication Safety layer wraps medication candidates in a dual-layer
DDI check using a local synthetic stub:

- `clear` — synthetic stub returned no interaction. Write may proceed
  through Truth Resolution.
- `warning` — possible interaction; write is held for operator
  confirmation.
- `blocked` — high-severity interaction; write is rejected and the
  candidate is held.
- `not_checked` / pending — DDI service was unavailable at evaluation
  time; the candidate is queued and **not** written.

The DDI stub is synthetic. No real PatientNotes API is called.

---

## 8. What "consensus status" means

When multiple connectors are queried, the Consensus Engine groups
extracted facts by `(specialty, fact_type, entity_text)` and assigns:

- `agreed` — multiple connectors return the same structured value.
- `single_source_penalized` — only one connector supplied the fact;
  confidence is reduced by 0.75.
- `contradicted` — connectors disagree on the structured value;
  routed to Truth Resolution as quarantine-only.
- `all_responses_discarded` — no successful responses; escalation
  flagged.

Consensus does **not** synthesize a new value across contradictions.
Consensus does **not** auto-write to the active tier.

---

## 9. What "coding unmapped" means

The Medical Coding layer attempts to map clinical entities to a
synthetic terminology source. Outcomes:

- `mapped` — entity was found in the local synthetic table.
- `unmapped` — no entry; the entity is left without a code.
- `ambiguous` — multiple candidates; the entity is left without a code
  pending operator decision.

The coder **does not invent codes**, does not promote hypothesis
facts on coding success, and does not clear DDI status.
