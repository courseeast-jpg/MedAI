# MedAI Final Operator Runbook

## Start MedAI

Normal local UI launcher:

```powershell
Start_MedAI_UI.bat
```

If the browser does not open automatically, open:

```text
http://localhost:8501
```

To stop MedAI, close the terminal window or press `Ctrl+C`.

## Encrypted Launcher

If using the encrypted runtime launcher, use the existing local launcher and key-management instructions already validated in the CKA security blocks. Keep key material outside git and avoid copying private runtime stores into reports.

## Final Validation Commands

```powershell
python -m pytest tests
python scripts/run_cka_final_mvp_release_validation.py
python scripts/run_b07_term01_opt_in_integration_validation.py
python scripts/run_medai_route_fix01_validation.py
python scripts/run_medai_terminology_sources_preflight.py
python scripts/run_medai_terminology_inventory.py --terminology-root terminology_data
```

## Terminology Source Preflight

```powershell
python scripts/run_medai_terminology_sources_preflight.py
```

Expected safe result: `terminology_sources_preflight_ready`.

This checks source folder shape and private acknowledgment presence only. It does not import terminology data and does not read private acknowledgment contents.

## Terminology Inventory

```powershell
python scripts/run_medai_terminology_inventory.py --terminology-root terminology_data
```

Expected safe result: `terminology_data_inventory_report_ready`.

This is aggregate metadata only. It does not print licensed terminology rows.

## B07-TERM Validation

```powershell
python scripts/run_b07_term01_opt_in_integration_validation.py
```

Expected safe result: `b07_term01_opt_in_integration_ready`.

B07 terminology metadata is:

- opt-in
- default-off
- hypothesis-only
- review-only
- not an authority source

Unknown terms remain unmapped. Ambiguous terms remain manual-review.

## ROUTE-FIX Validation

```powershell
python scripts/run_medai_route_fix01_validation.py
```

Expected safe result: `medai_route_fix01_ready`.

## Private Terminology File Discipline

Keep these out of git:

- private terminology source folders
- private license acknowledgment files
- runtime terminology stores
- source terminology release files
- database and key files
- PHI-bearing PDFs, images, text files, and archives

Before any commit, run:

```powershell
git diff --cached --name-only
git status --short
```

Only public-safe code, tests, docs, and reports should be staged.
