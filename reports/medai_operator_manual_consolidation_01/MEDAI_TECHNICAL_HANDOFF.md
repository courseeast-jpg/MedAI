# MedAI — Technical Handoff

Maintainer/developer-oriented summary consolidated from the public-safe
operator-readiness chain (`RELEASE-HANDOFF-01`, `OPERATOR-UAT-01`,
`REAL-WORLD-OPERATOR-UAT-02`, `PACKAGING-LAUNCHER-HARDEN-01`,
`UI-USABILITY-POLISH-01`, `DATA-RUNTIME-HARDEN-01`, `PARK-26`,
`ROADMAP-03`). Sibling to `MEDAI_LOCAL_OPERATOR_MANUAL.md`. Use this
document when picking the project up from a clean checkout to
understand what is shipped, what is parked, what to validate, and what
not to touch.

## 1. Current branch / state summary

| Field | Value |
| --- | --- |
| Repo | `courseeast-jpg/MedAI` |
| Branch | `clinical-knowledge-architecture` |
| HEAD (short, at time of this handoff) | `e378f8c` (ROADMAP-03 receipt refresh; previous commit `86a3b73` is ROADMAP-03 itself) |
| Latest parking snapshot | `PARK-26` at `91b9eba` |
| Local operator release | **ready** (no blocking bugs) |
| Cumulative `runtime_behavior_changed` across recent blocks | False |
| Cumulative `cue_expansion_recommended` | False |
| `local_only_posture_preserved` | True |
| `external_api_used` | False |
| `external_api_enabled` | False |
| Whole MedAI project | **~93.2%** done / ~6.8% remaining |

## 2. Key parked tracks and tag map

Two annotated tags accompany each parking commit. They must not be
moved, deleted, or recreated. PARK-24/25/26 are commits with no
parking tags pushed at this stage (the corresponding parking blocks
intentionally deferred tag pushes); the commits themselves are
discoverable by `git log` and by the PARK report directories under
`reports/`.

| Track / parking block | Parking commit (short) | Tag pair |
| --- | :-: | --- |
| Residual-Unknown — language metadata | `3e46461` | `medai-unknown-diag-language-metadata-ready-2026-05-19`, `medai-final-parked-post-unknown-diag-language-metadata-2026-05-19` |
| Text-layer evaluation spec | `9f9e22d` | `medai-text-layer-eval-spec-ready-2026-05-19`, `medai-final-parked-post-diag-16-2026-05-19` |
| PDF text/layout quality default-off | `f4d3cc6` | `medai-pdf-text-layout-quality-default-off-ready-2026-05-19`, `medai-final-parked-post-diag-18-2026-05-19` |
| PDF text/layout Streamlit wiring | `748c32a` | `medai-pdf-text-layout-quality-streamlit-wiring-ready-2026-05-19`, `medai-final-parked-post-diag-19-2026-05-19` |
| DIAG-20 corpus-side env-on UAT | `1b14ffe` | _(untagged commit; report dir: `reports/medai_park_24_diag20_operator_uat/`)_ |
| DIAG-21 Streamlit fixture audit | `6b31678` | _(untagged commit; report dir: `reports/medai_park_25_diag21_streamlit_fixture_audit/`)_ |
| Operator readiness + runtime hardening | `91b9eba` | _(untagged commit; report dir: `reports/medai_park_26_operator_runtime_readiness/`)_ |

The four pre-existing tag pairs above (PARK-20..23) are the **only**
tags that should be on the parking commits at this point. Must not
add new tags to the PARK-24, PARK-25, or PARK-26 commits without
explicit approval.

## 3. Validation command map

The shipped "operator-ready" definition is the union of these five
reports-only validations. Each is idempotent and safe to run
repeatedly:

| Command | Output report directory | Expected conclusion |
| --- | --- | --- |
| `python scripts/run_cka_final_mvp_release_validation.py` | `reports/cka_final_mvp_release/` | `cka_mvp_release_package_ready` (12/12 validation cases; 693 tests; `external_api_used: false`) |
| `python scripts/run_b07_term01_opt_in_integration_validation.py` | `reports/b07_term01_opt_in_integration/` | `cases_failed: 0`, `external_api_used: false` |
| `python scripts/run_medai_route_fix01_validation.py` | `reports/medai_route_fix_01/` | `medai_route_fix01_ready`, `passed: true` |
| `python scripts/run_medai_ui_ops_panel_validation.py` | `reports/medai_ui_ops_01/` | `medai_ui_ops_panel_ready` |
| `python scripts/run_medai_ui_boot_fix_validation.py` | `reports/medai_ui_boot_fix_01/` | `medai_ui_boot_fix_startup_resilience_ready` |

For privacy of generated public reports, use:

```python
from clinical_knowledge.privacy import check_public_report_payload
```

Every public-safe report in `reports/` is expected to pass this check.
A failure indicates either a long hex string interpreted as a secret
(use short SHAs, max ~7 chars), a medical-looking filename literal,
or genuine leakage that must be redacted before commit.

## 4. Safety / privacy invariants

These invariants are enforced across every block in the chain and
must not regress without a new approved planning block:

- `runtime_behavior_changed`: False (cumulative across the recent
  block chain).
- `app_main_modified`: False (in reports-only blocks).
- `launcher_files_modified`: False.
- `startup_preflight_modified`: False.
- `config_modified`: False.
- `extraction_behavior_changed`: False.
- `ocr_behavior_changed`: False.
- `classifier_behavior_changed`: False.
- `threshold_behavior_changed`: False.
- `cue_expansion_recommended`: False.
- `cue_expansion_performed`: False.
- `external_api_used`: False.
- `external_api_enabled`: False.
- `source_documents_opened`: False.
- `private_files_opened`: False.
- `runtime_db_contents_opened`: False.
- `raw_text_printed` / `raw_filenames_printed` / `private_paths_printed`
  / `secrets_printed`: False.
- `clinical_value_parsing_performed`: False.
- `clinical_interpretation_performed`: False.
- `diagnosis_inference_performed` / `medication_inference_performed`
  / `ddi_inference_performed` / `treatment_inference_performed`
  / `abbreviation_expansion_performed`: False.
- `accepted_count` / `auto_accept_allowed_count`
  / `external_api_used_count`: 0.
- `all_records_review_bound`: True.

## 5. Environment and launcher notes

Local-only environment defaults (set by the launchers):

| Env var | Default | Source |
| --- | :-: | --- |
| `MEDAI_LOCAL_ONLY` | `1` | `PACKAGING-LAUNCHER-HARDEN-01` |
| `MEDAI_ALLOW_EXTERNAL_API` | `0` | `PACKAGING-LAUNCHER-HARDEN-01` |
| `MEDAI_REQUIRE_PII_SCRUB` | `1` | `PACKAGING-LAUNCHER-HARDEN-01` |
| `MEDAI_PRIVACY_AUDIT` | `1` | `PACKAGING-LAUNCHER-HARDEN-01` |

Shipped launchers (in repo root):

- `Start_MedAI_UI.bat` — primary operator launcher (default).
- `Start_MedAI_UI_Silent.vbs` — console-less variant of the .bat.
- `Start_MedAI_Test_UI.bat` — UI variant used by the UI validation
  receipts.
- `Start_MedAI_UI_Encrypted.bat` — encrypted-packaging variant.

Each launcher `cd`s to the script directory before launching Streamlit
(`DATA-RUNTIME-HARDEN-01: launcher_path_assumptions = ready`). The
startup preflight (`app/startup_preflight.py`) emits metadata-only
diagnostics; it never inspects runtime DB row contents.

The two env vars introduced by the residual-Unknown chain remain
default-off and are documented for completeness — they must stay off
in production:

- `MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_IMPL_ENABLED` (DIAG-17;
  emits read-only metadata when truthy)
- `MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_UI_ENABLED` (DIAG-18;
  wires the read-only render plan into the Advanced technical
  details expander when both env vars are truthy)

Both are gated by the DIAG-18 helper. With either env var unset or
falsy, the DIAG-19 Streamlit wiring block executes zero `st.*` calls.

## 6. Known sandbox limitations

These are CI / development-host limitations, not project bugs. They
have been observed across multiple blocks in this chain.

- **Full repo-wide pytest is blocked in some sandboxes.** Tests that
  import Streamlit at module load (≈ 30 tests under `tests/`) fail
  collection when Streamlit is not installed in the test environment.
  These tests pass on a developer machine that has Streamlit. The
  documented workaround for reports-only blocks is to run the
  Streamlit-free subsets and the five fixed validation scripts in
  Section 3.
- **Local push proxy may be intermittently unreachable.** When the
  proxy is down, push via the github-direct route
  (`https://github.com/courseeast-jpg/MedAI.git`) using the
  `gh`-installed credentials. After the push, remove the temporary
  remote.
- **Tag push via the proxy historically returns HTTP 403 on
  `refs/tags/*` writes.** All tag pushes in this chain (`PARK-20..23`)
  went via the github-direct route. PARK-24/25/26 commits were
  intentionally left untagged.
- **Public-report privacy scanner treats long hex strings as
  potential secrets.** Use short SHAs (max ~7 chars) in any
  human-readable field of a public report; reserve full 40-char
  SHAs for git tooling only.

## 7. What not to reopen without approval

This list intentionally mirrors `RELEASE-HANDOFF-01.do_not_reopen_
without_approval`, augmented with later blocks. Do **not** reopen any
of these without a new approved planning block:

- OCR routing changes.
- PDF text / layout / table extraction behavior changes.
- Classifier / threshold / scoring changes.
- **Cue pack expansion (explicitly NOT recommended).**
- Lab-value parsing.
- Medication or DDI parsing.
- Diagnosis / treatment / DDI / abbreviation inference.
- External API enablement.
- Parking tag changes (move, delete, or recreate).
- `app/main.py`, the launchers, `app/startup_preflight.py`,
  `app/config.py` — these are part of the parked operator surface.
- Residual-Unknown micro-diagnostics (track is parked at ~99.99%
  closed; reopen only on a fresh operator-UAT signal).

## 8. Recommended next sequence

From `ROADMAP-03.top_recommended_next_phase`:

1. **MEDAI-OPERATOR-MANUAL-CONSOLIDATION-01** — this block.
2. **MEDAI-PACKAGING-DEPLOYMENT-POLISH-03** — launcher / install /
   first-run UX polish that references this consolidated manual.
3. **MEDAI-FREEZE-LOCAL-OPERATOR-RELEASE** — final release snapshot
   of the local operator artifact.
4. **MEDAI-ROADMAP-04** before any new expansion block (e.g. v2
   architecture, terminology integration, real-corpus validation).

After step 3, the local operator release becomes the durable shipped
artifact. Any v2 work, terminology integration, or real-corpus
validation must originate from a fresh ROADMAP block; must not pick
those items off this document.

## 9. Deferred blocks

The following are deferred and must not be reopened without an
explicit new approved planning block:

- **MORE-UNKNOWN-DIAGNOSTICS** — deferred. The residual-Unknown
  reduction track is parked at ~99.99% closed. Reopen only on a
  fresh operator-UAT signal.
- **CUE-EXPANSION** — explicitly **NOT** recommended. Standing
  project posture across `DIAG-13..21` and `ROADMAP-01..03`.
- **Broad v2 architecture** — deferred until after
  `FREEZE-LOCAL-OPERATOR-RELEASE` and a dedicated
  `V2-ARCHITECTURE-SPEC` block.

## 10. Maintenance notes

- All public reports live under `reports/`. Each public report
  directory typically contains a `<phase>_report` JSON file, a
  `<phase>_report` Markdown file, and a short summary `MEDAI_*.md`.
- Existing repo docs (`README.md`, `RELEASE_OPERATOR_GUIDE.md`,
  `RELEASE_QUICKSTART_LOCAL_ONLY.md`,
  `RELEASE_LIMITATIONS_AND_SAFETY.md`, `ARCHITECTURE.md`) are not
  modified by this block. They remain authoritative for their
  original topics; this handoff is the **navigation index** into
  them and into the more recent operator-readiness reports.
- When generating new public reports, use the existing
  `clinical_knowledge.privacy.check_public_report_payload` to gate
  commits. Use short SHAs in human-readable fields.
- Branch pushes should go to `origin` when the proxy is healthy.
  When unhealthy, use the github-direct route and remove the
  temporary remote after the push.
- Must not run `git push --tags`. Must not use `--force` /
  `--force-with-lease`. Tag pushes must be explicit by tag name.

---

_This handoff is consolidated from public-safe reports only. No
source documents, raw OCR text, raw document text, raw filenames,
private paths, PHI, secrets, DBs, backups, bundles, keys, or
licensed terminology data were read or referenced._
