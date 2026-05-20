# MedAI — Local Operator Manual

Consolidated from the public-safe operator-readiness chain (`RELEASE-
HANDOFF-01`, `OPERATOR-UAT-01`, `REAL-WORLD-OPERATOR-UAT-02`,
`PACKAGING-LAUNCHER-HARDEN-01`, `UI-USABILITY-POLISH-01`,
`DATA-RUNTIME-HARDEN-01`, `PARK-26`). This is the single document an
operator should read to start, verify, and use MedAI locally. It does
not replace clinical judgment; every result must be reviewed by a
human.

## 1. What MedAI is in the current local release

MedAI is a **local-only, human-review-bound** document-processing app
for medical records. In the current release it:

- Runs entirely on the operator's machine. By default no data leaves
  the local environment.
- Ingests PDFs (text and image), TXT and other supported file types
  through a **Run & Review** workflow inside a Streamlit UI.
- Produces per-document **result cards** that classify each record into
  workflow buckets such as `accepted`, `review`, `review_ocr_quality`,
  `empty`. These are **workflow labels, not clinical acceptance**.
- Surfaces optional read-only metadata inside collapsed **Advanced
  technical details** sections. Advanced details are default-safe: they
  emit no buttons, no actions, and no clinical inference.
- Does **not** perform clinical interpretation, diagnosis inference,
  medication inference, DDI inference, treatment inference,
  abbreviation expansion, lab-value parsing, or auto-acceptance.
- Does **not** call any external API by default. `MEDAI_LOCAL_ONLY=1`
  and `MEDAI_ALLOW_EXTERNAL_API=0` are the shipped defaults.

## 2. How to start the app locally

1. Open the repository folder on the operator machine.
2. Double-click **`Start_MedAI_UI.bat`** (Windows). The launcher will:
   - `cd` to the repository directory before starting the UI.
   - Apply the local-only environment defaults
     (`MEDAI_LOCAL_ONLY=1`, `MEDAI_ALLOW_EXTERNAL_API=0`,
     `MEDAI_REQUIRE_PII_SCRUB=1`, `MEDAI_PRIVACY_AUDIT=1`).
   - Start Streamlit on http://localhost:8501.
3. If the browser does not open automatically, manually open
   <http://localhost:8501>.
4. Confirm the **Run & Review** tab is visible.

Alternative launchers:

- `Start_MedAI_UI_Silent.vbs` — same as the .bat but without a console
  window.
- `Start_MedAI_Test_UI.bat` — launches the UI variant used by the UI
  validation receipts; safe to run locally.
- `Start_MedAI_UI_Encrypted.bat` — launches the UI in the encrypted
  packaging variant; use only on machines configured for encrypted
  storage.

> If the launcher closes immediately without opening a browser, see
> Section 9 (Common startup / recovery guidance).

## 3. How to verify health and readiness

Open a terminal in the repository folder and run any of the fixed
validation commands below. Each command is reports-only and safe to
run repeatedly:

```text
python scripts/run_medai_ui_boot_fix_validation.py
python scripts/run_medai_ui_ops_panel_validation.py
python scripts/run_cka_final_mvp_release_validation.py
python scripts/run_b07_term01_opt_in_integration_validation.py
python scripts/run_medai_route_fix01_validation.py
```

A healthy run prints a `conclusion: ..._ready` line and writes a
report into `reports/`. The five commands above are the **operator
health-check set**; they are the same commands referenced in
`RELEASE-HANDOFF-01`. A successful run of all five — `cases_failed: 0`
on B07, `passed: True` on ROUTE-FIX, `_ready` conclusions on UI ops /
UI boot, and `693 passed` on the final CKA MVP validation — is the
current "operator ready" definition.

If any command prints a non-`_ready` conclusion, see Section 11 (when
to stop and ask for engineering help).

## 4. Using Run & Review at a high level

1. **Current Run** tab: upload one or more supported files (PDF, TXT,
   RTF, TIF/TIFF, PNG, JPG/JPEG, BMP, WEBP). MedAI processes them
   locally.
2. Each processed item appears as a **result card** with a workflow
   label such as `accepted`, `review`, `review_ocr_quality`, or
   `empty`.
3. A short operator-summary caption appears on each card explaining
   what the label means in workflow terms — **not** in clinical
   terms.
4. Open a card's **Advanced technical details** expander only when you
   need diagnostic context. The contents are read-only and carry an
   explicit "review required" disclaimer; they never accept, reject,
   approve, or change any document.
5. The numeric run counters (e.g. `accepted: N`) are workflow tallies,
   not clinical acceptance counts. They will not match clinical
   adjudication.

Other tabs:

- **Blind Audit** — runs the blind generalization audit. Inputs and
  outputs are local; the audit is reports-only.
- **Report Archive** — browses previous public-safe reports.
- **Review Package** — auto-grouped buckets that explain pending
  review cases without requiring the operator to open every document
  individually.

## 5. What "review-bound" means

Every result MedAI emits is **bound to human review**. Concretely:

- `accepted_count` is a workflow count, not a clinical acceptance
  count. The operator-readiness chain validates that
  `accepted_count` does not change as a side-effect of any UI or
  metadata block.
- `auto_accept_allowed` is **always False** across every emitted
  metadata / render-plan record (`auto_accept_allowed_count: 0`).
- Every result card is review-bound (`all_records_review_bound:
  true`).
- Advanced technical details add no acceptance, rejection, mutation,
  callback, button, form, or data-layer write. They are read-only.

Operationally: even a card labeled `accepted` must be **spot-checked
against the source document by a human** before the data is used
downstream. The label means "passed the local workflow heuristic",
not "clinically validated".

## 6. What "local-only" means

The shipped configuration is local-only:

| Env var | Default | Meaning |
| --- | :-: | --- |
| `MEDAI_LOCAL_ONLY` | `1` | Restricts the app to local processing. |
| `MEDAI_ALLOW_EXTERNAL_API` | `0` | External API surface disabled. |
| `MEDAI_REQUIRE_PII_SCRUB` | `1` | PII scrubbing required before any export. |
| `MEDAI_PRIVACY_AUDIT` | `1` | Privacy audit enabled on report output. |

Concretely the local-only posture implies:

- No outbound HTTP to LLM, terminology, or coding services by default.
- No telemetry by default.
- No background sync of documents to any remote location.
- Reports written to `reports/` are public-safe (PHI-scrubbed); raw
  OCR text, raw document text, raw filenames, and private paths are
  not embedded in those reports.

If a downstream phase ever asks the operator to flip
`MEDAI_ALLOW_EXTERNAL_API=1`, treat that as an **escalation** and
verify the request is from authorized engineering before proceeding.

## 7. What "Advanced technical details" are and are not

Advanced technical details are **collapsed by default** on every
result card. Their entire purpose is to give the operator diagnostic
context when a card looks unusual.

Advanced details **are**:

- Read-only.
- Default-safe (collapsed; no widget input).
- Limited to safe Streamlit primitives (`st.markdown`, `st.caption`).
- Bound to controlled-vocabulary text only; no raw extracted text,
  no raw filenames, no private paths, no PHI, no secrets are
  rendered.
- Carrying an explicit "review required" disclaimer.

Advanced details **are not**:

- A place to accept, reject, approve, or override anything.
- A clinical interpretation surface.
- A trigger for any downstream write.
- A diagnostic free-text field that prints raw document content.

If you ever see a button, form, accept/reject action, or raw
document text inside Advanced technical details, **stop and
escalate** — that would indicate a regression beyond the current
release.

## 8. What the operator must avoid

Operator restrictions — must **not**:

- Edit `app/main.py`, the launchers, `app/startup_preflight.py`, or
  `app/config.py` to "tune" the system. These files are part of the
  parked operator-readiness surface.
- Set `MEDAI_ALLOW_EXTERNAL_API=1` or any external-API toggle on
  your own.
- Move, delete, or recreate any parking tag (`medai-*-2026-05-*`).
- Paste source-document content, filenames, or private paths into
  any report, screenshot, ticket, chat, or email. Always use
  anonymized identifiers (e.g. `record_001`).
- Treat a `accepted` workflow label as clinical acceptance.
- Re-enable OCR/extraction/classifier/threshold/cue/lab-value/
  medication/DDI/abbreviation behavior changes locally. These are
  parked tracks (see Section 10).
- Add cue packs. Cue expansion is explicitly **NOT** recommended
  across the entire project.

## 9. Common startup / recovery guidance

| Symptom | Likely cause | Safe next step |
| --- | --- | --- |
| Launcher window closes immediately | Working directory not the repo, or Python not on PATH | Open a terminal, `cd` into the repo, run `python scripts/run_medai_ui_boot_fix_validation.py`. Read the report it writes. |
| Browser does not open | Default browser not configured | Open <http://localhost:8501> manually. |
| Streamlit page is blank or 500s | UI boot regression | Run `python scripts/run_medai_ui_boot_fix_validation.py` and `python scripts/run_medai_ui_ops_panel_validation.py`. If either prints a non-`_ready` conclusion, escalate. |
| Validation command prints an error | Stale dependency or partial install | Re-run the same command. If it fails twice with the same error, escalate. |
| Run & Review missing | UI tab regression | Run the UI ops + UI boot validations. Confirm `medai_ui_ops_panel_ready` and `medai_ui_boot_fix_startup_resilience_ready`. |
| Card shows raw text or filename | Privacy regression | **Stop. Must not screenshot.** Escalate immediately. |
| Card has accept/reject buttons in Advanced technical details | Read-only regression | **Stop.** Escalate immediately. |

The launcher already `cd`s to the script directory before startup
(`PACKAGING-LAUNCHER-HARDEN-01`), the startup preflight emits
metadata-only diagnostics (`DATA-RUNTIME-HARDEN-01`), and the
DB-availability probe is read-only. Standard recovery is: read the
preflight output, run the five health-check validations, and
escalate if any remains non-`_ready`.

## 10. Current parked tracks

The following tracks are deliberately **parked** at this release.
They must not be reopened by the operator. Reopening any of them
requires a new approved ROADMAP block.

- Residual-Unknown reduction track (`PARK-20`, `PARK-21`, `PARK-22`,
  `PARK-23`, plus untagged `PARK-24/25` snapshots).
- PDF text / layout quality track (`PARK-22`, `PARK-23`).
- DIAG-20 corpus-side env-on operator UAT (`PARK-24`).
- DIAG-21 Streamlit fixture audit (`PARK-25`).
- Operator readiness + runtime hardening (`PARK-26`).

Behavior tracks that must **not** be reopened without engineering
approval:

- OCR routing or OCR engine behavior.
- Default PDF text extraction or default layout/table extraction.
- Classifier behavior, thresholds, or scoring.
- Cue-pack expansion (explicitly **NOT** recommended).
- Lab-value parsing.
- Medication or DDI parsing.
- Diagnosis / treatment / DDI / abbreviation inference.
- External API enablement.
- Parking tag changes.

## 11. When to stop and ask for engineering help

Stop using MedAI and escalate to engineering if any of the following
occurs:

1. Any of the five health-check validation commands prints a
   non-`_ready` conclusion two runs in a row.
2. Raw document text, raw filenames, private paths, PHI, or secrets
   appear anywhere in the UI or in a public report.
3. A result card grows buttons, forms, accept/reject controls, or any
   interactive widget inside Advanced technical details.
4. `accepted_count` changes as a side-effect of opening a card,
   scrolling, or clicking inside Advanced technical details.
5. The UI prompts you to enable an external API, set
   `MEDAI_ALLOW_EXTERNAL_API=1`, or send data off-machine.
6. The launcher window keeps closing immediately and the UI boot
   validation never reaches `_ready`.
7. A parking tag appears moved, deleted, or recreated.
8. Any clinical decision is being made directly from a workflow
   label without a human reviewing the source document.

For escalation, share the most recent validation report directory
paths under `reports/` (e.g.
`reports/medai_ui_boot_fix_01/`). Must **not** share source
documents, raw text, raw filenames, private paths, or screenshots
that contain any of those.

---

_This manual is consolidated from the public-safe reports in
`reports/`. It does not replace clinical judgment. Every result
remains human-review-bound._
