# MedAI Vertex Decision Audit — Operator Handoff Guide

- **Document ID:** MEDAI-VERTEX-SEMANTIC-DECISION-AUDIT-OPERATOR-HANDOFF-DOC-15W
- **Status:** Active operator handoff / governance guide (documentation only).
- **Applies to:** The read-only "Vertex Decision Audit" operator tab and its exports.

This guide is documentation only. It changes no code, no UI, no decision store, and
makes no provider call.

## 1. Purpose

The **Vertex Decision Audit** tab lets an operator inspect the review-bound Vertex
semantic extraction decisions and their exports. It is a read-only audit surface: it
shows which synthetic Vertex semantic findings were accepted-for-review, rejected, or
deferred, with full provider provenance, evidence anchors, unknown values, and
uncertainty flags — and it offers JSON / CSV / markdown exports for offline review.

It does **not** create active MKB records, does **not** auto-accept anything, and does
**not** call any provider. It exists so a human can audit the recorded decisions safely.

## 2. How to reach the panel

1. Open the MedAI app.
2. In the sidebar, enable **"Show advanced tools"** (advanced/operator tabs).
3. Select the **"Vertex Decision Audit"** tab.

## 3. What the operator should see

- Panel title: **Vertex Semantic Review Decision Audit**.
- A **no-live / read-only indicator**: "Read-only audit. No live provider call. No active MKB records are created."
- **Provider route/model:** `vertex` / `gemini-2.5-flash-lite`.
- **Decision counts:** total **15**, accepted_for_review **13**, rejected **1**, deferred **1**.
- **Package-family breakdown:**
  - `portal_result_cards`: 4
  - `cytology_pathology_narrative`: 3
  - `urinalysis_table_like_lab`: 4
  - `mixed_narrative_numeric_result`: 4
- **Safety status:** active writes **0**, active MKB records **0**, `auto_accept` **false**, `review_required` **true**.
- **Evidence/provenance rows:** per decision — evidence anchor, source evidence text, source report reference, audit reason, and provider route/model.
- **Unknown values** shown explicitly.
- **Uncertainty flags** shown.
- **Hallucinated field count: 0**.

## 4. Export instructions

The panel offers read-only download affordances for:

- **JSON export** — `reports/medai_vertex_semantic_review_decision_audit_export_15s/decision_audit_export.json`
- **CSV export** — `reports/medai_vertex_semantic_review_decision_audit_export_15s/decision_audit_export.csv`
- **Markdown audit summary** — `reports/medai_vertex_semantic_review_decision_audit_export_15s/decision_audit_summary.md`

**Exports are read-only. No active MKB records are created.** Downloading an export does
not change any decision status and does not mutate the decision store.

## 5. Governance guarantees

- **Review-bound only** — every decision stays a review draft; nothing is promoted to active knowledge.
- **No active MKB write** — the panel and exports never write active MKB records (`active_written_count = 0`).
- **No auto-accept** — `auto_accept` is always false; a human decision is always required.
- **No provider/network call** — the panel replays recorded artifacts; it never calls Vertex/Gemini or any provider.
- **Decision store remains unchanged** during audit/export (fingerprint verified unchanged in 15S/15T/15U/15V).
- **Recorded synthetic/redacted artifacts only** — all content originates from synthetic, redacted fixtures.
- **Not yet cleared for unrestricted real medical documents** — this chain is validated on synthetic/redacted inputs only.

## 6. Operator do / don't

| Do | Don't |
| --- | --- |
| Inspect decisions | Treat Vertex findings as active clinical facts |
| Verify evidence anchors | Assume real-document clearance |
| Check unknown / uncertainty flags | Bypass human review |
| Download exports for review | Edit active MKB from this panel |
| Escalate suspicious or unsupported findings | Run live provider scripts from the UI |

## 7. Troubleshooting

Each of the following is a **safety failure — stop and report** (do not work around it):

- **Tab missing** — the "Vertex Decision Audit" tab is not present under advanced tools. Stop and report as safety failure.
- **Export missing** — a JSON/CSV/markdown export artifact is absent. Stop and report as safety failure.
- **Decision counts mismatch** — totals differ from total 15 / accepted_for_review 13 / rejected 1 / deferred 1. Stop and report as safety failure.
- **Read-only statement missing** — "Exports are read-only. No active MKB records are created." is not shown. Stop and report as safety failure.
- **Any active-write counter not zero** — `active_written_count` or `active_mkb_record_created_count` is non-zero. Stop and report as safety failure.
- **Any provider/live indicator unexpectedly true** — `live_call_made` or `external_api_used` is true. Stop and report as safety failure.

## 8. Evidence chain

| Block | Scope | Result |
| --- | --- | --- |
| 15P-C | Portal result-card live Vertex comparison | PASS |
| 15P-D | Remaining three families live Vertex comparison | PASS |
| 15Q | Operator review surface | PASS |
| 15R | Review-bound decision persistence | PASS |
| 15S | Read-only audit/export | PASS |
| 15T | Read-only operator panel | PASS |
| 15U | Tab navigation wiring | PASS |
| 15V | Operator UAT | PASS |

## 9. Current limits

- Synthetic / redacted fixtures only.
- No unrestricted real medical document route yet.
- No active MKB write from Vertex findings.
- No auto-promotion of any finding.
- Billing visibility is still pending/rounding, but it is not relevant to this no-live handoff.

## 10. Next stage preview

- A bounded Vertex calibration batch on synthetic/redacted fixtures.
- An explicit token/cost ledger.
- Still **no real-document clearance** until privacy stripping and operator gates are proven for real-document routing.
