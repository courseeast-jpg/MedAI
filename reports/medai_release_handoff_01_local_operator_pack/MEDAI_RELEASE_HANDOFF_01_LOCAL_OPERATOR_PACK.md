# MEDAI-RELEASE-HANDOFF-01

## Executive status

MedAI is ready for a local operator handoff pack. The current state is local-only, review-bound, and validated through OPERATOR-UAT-01 and PACKAGING-LAUNCHER-HARDEN-01. This handoff changes no runtime code and does not modify launchers.

## Operator quick-start

1. Open a terminal in the MedAI repository.
2. Run `Start_MedAI_UI.bat`.
3. If the browser does not open, go to `http://localhost:8501`.
4. Confirm the Run & Review workflow is available.
5. Keep every document in human review until a qualified operator manually checks the source document.

## Health check and validation commands

Use these local validation commands when confirming readiness:

```powershell
python scripts/run_medai_ui_boot_fix_validation.py
python scripts/run_medai_ui_ops_panel_validation.py
python scripts/run_cka_final_mvp_release_validation.py
python scripts/run_b07_term01_opt_in_integration_validation.py
python scripts/run_medai_route_fix01_validation.py
```

Expected results:

- UI boot validation reports startup resilience ready.
- UI ops validation reports the fixed allowlist panel ready.
- Final CKA MVP validation reports 12/12 cases and 693 tests.
- B07 validation reports 6/6 cases and remains default-off/hypothesis-only.
- ROUTE-FIX validation reports ready.

## Local-only and external API posture

The normal launcher sets local-only defaults:

- `MEDAI_LOCAL_ONLY=1`
- `MEDAI_ALLOW_EXTERNAL_API=0`
- `MEDAI_REQUIRE_PII_SCRUB=1`
- `MEDAI_PRIVACY_AUDIT=1`

External APIs remain disabled. Cloud tools are not part of this handoff.

## What is parked

- PDF text/layout quality diagnostics are parked through DIAG-21 and PARK-25.
- Residual Unknown micro-diagnostics are parked and should not be reopened without a specific operator need.
- Cue expansion remains explicitly not recommended.

## What not to reopen without approval

Avoid reopening these areas without approval:

- OCR routing changes.
- PDF text/layout extraction behavior.
- Document classifier thresholds or scoring.
- Cue pack expansion.
- Lab value parsing.
- Medication, dose, frequency, duration, or DDI parsing.
- External API enablement.
- PARK tag changes.

## Common failure recovery guidance

- If the browser does not open, manually open `http://localhost:8501`.
- If startup fails before the UI renders, run UI boot validation and use the startup diagnostics guidance.
- If operator controls look unavailable, run UI ops validation.
- If a validation script rewrites report receipts during a packaging block, restore generated validation deltas before staging handoff files.
- Avoid deleting databases, keys, private source files, or runtime stores as a troubleshooting shortcut.

## Safety and privacy invariants

- No diagnosis is performed.
- No clinical interpretation is added.
- No lab values are accepted automatically.
- No medication instructions are interpreted.
- No external API is enabled.
- Private/source documents, raw OCR text, raw document text, filenames, private paths, PHI, secrets, DBs, backups, bundles, keys, and terminology data stay out of public reports and git staging.

## Validation evidence

- OPERATOR-UAT-01: local operator workflow smoke receipt ready.
- PACKAGING-LAUNCHER-HARDEN-01: local launch readiness ready.
- ROADMAP-01: next workstreams ranked; cue expansion not recommended.
- PARK-25: Streamlit fixture audit parked; default-off behavior confirmed.
- UI boot, UI ops, final CKA MVP, B07, and ROUTE-FIX validations passed.

## Next strategic options

1. Broader MedAI roadmap implementation.
2. Optional packaging polish if a more formal install/runbook bundle is needed.
3. Optional real-world operator UAT using safe receipt-only reporting.

Cue expansion remains NOT recommended.
