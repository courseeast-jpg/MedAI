# OPERATOR START HERE — MedAI CKA MVP Scaffold

This is the one-page operator entry point for the Clinical Knowledge
Architecture MVP scaffold. Read it before launching anything.

---

## Critical safety warnings — read first

The CKA MVP scaffold:

- **is NOT production-autonomous**
- **is NOT a medical device**
- **does NOT issue diagnoses**
- **does NOT prescribe medications**
- **does NOT provide clinical advice or dosing**
- **has NO real external connectors active** — all connectors are
  local synthetic stubs

Any real medical interpretation requires verification by a qualified
medical professional. The scaffold is technical safety infrastructure
only.

---

## 1. One-click start (Windows)

Double-click or run from a terminal:

```
Start_MedAI_UI.bat
```

The launcher sets all required local-only safety flags:

- `MEDAI_LOCAL_ONLY=1`
- `MEDAI_ALLOW_EXTERNAL_API=0`
- `MEDAI_REQUIRE_PII_SCRUB=1`
- `MEDAI_PRIVACY_AUDIT=1`

It then starts Streamlit on `http://localhost:8501` and opens the
browser automatically. To stop, close the terminal window or press
`Ctrl+C`.

## 2. Final validation (any platform)

To re-confirm the MVP scaffold is healthy, run:

```
python scripts/run_cka_final_mvp_release_validation.py
```

A passing run reports:

- `12/12 cases passed`
- `26/26 preflight checks passed`
- `Total tests passed: 693` (or higher)
- `External API used: False`
- `Production autonomous: False`
- `Frozen HITL release reopened: False`

If any of those flip from the values above, **stop using the scaffold
and investigate** before any further operator action.

---

## 3. What to check in the UI

After the browser opens at `localhost:8501`, the operator app shows
five tabs:

| Tab | What to confirm |
|---|---|
| Current Run | Standard operator view — unchanged by the CKA scaffold |
| Blind Audit | Standard audit view — unchanged by the CKA scaffold |
| Report Archive | Existing archive — unchanged by the CKA scaffold |
| **Review Package** | The frozen HITL operator review package; should still load and display the existing release artifacts |
| **Clinical Knowledge Safety** | The new CKA panel (5th tab). Verify all nine sub-panels render: **MKB Status**, **Decision Engine**, **Privacy**, **Truth Resolution**, **Medication Safety**, **Enrichment / Hypothesis**, **Medical Coding**, **Multi-Connector Consensus**, **Release Readiness** |

In the **Clinical Knowledge Safety** tab, verify that the safety
banner / status chips show:

- **Local-only mode ON**
- **External APIs disabled**
- **Privacy audit ON**
- **All blocks loaded** (B01 through B10)
- **No clinical recommendations generated**
- **No prescription dosing advice generated**
- **Production autonomous: false**
- **Frozen HITL release reopened: false**

If any of those banners read otherwise, **stop and investigate**
before continuing.

---

## 4. What this scaffold is *not* for

Do not use this scaffold for:

- patient diagnosis
- prescribing or dose selection
- medication interaction decisions in real care
- replacement of a clinician's review
- generation of medical advice in any form
- any production-critical clinical workflow

These uses are out of scope for the MVP scaffold and would require a
separately-scoped, separately-approved roadmap track with its own
safety review.

---

## 5. Where to read more

Inside `reports/cka_final_mvp_release/`:

- `CKA_OPERATOR_GUIDE.md` — full operator guide (preflight, panels, Safe Mode, hypothesis tier, quarantine, DDI states, consensus, coding)
- `CKA_LIMITATIONS_AND_SAFETY.md` — binding scope statement
- `CKA_ARCHITECTURE_MANIFEST.md` — block / module / commit manifest (B01-B11)
- `CKA_CONTINUATION_SNAPSHOT.md` — handoff snapshot for the next contributor
- `cka_final_mvp_release_report.md` — final validation summary

Inside `reports/cka_operator_release_seal/`:

- `cka_operator_release_seal_report.md` — this seal block's full report
- `cka_operator_release_seal_report.json` — machine-readable seal status
- `OPERATOR_START_HERE.md` — this page
