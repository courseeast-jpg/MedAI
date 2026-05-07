# CKA-TERM-02 Operator Runbook

This runbook prepares the operator for a future local-only TERM-02 run. It does not run TERM-02.

## Manual Preparation

1. Obtain licensed terminology files from their official vendor portals outside MedAI.
2. Put the files under the local `terminology_data/` intake tree prepared by TERM-01A.
3. Create the private `LICENSE_ACK_PRIVATE.json` file locally after reviewing the applicable licenses.
4. Keep private acknowledgments and vendor files out of git.

## One-Command Readiness Pack

Run:

```powershell
python scripts/cka_terminology_manual_return_pack.py
```

Expected result:

- Intake folders exist.
- Readiness checks run.
- Dry-run planner runs.
- TERM-02 preflight gate runs.
- No import occurs.

## Individual Operator Commands

Prepare intake:

```powershell
python scripts/cka_terminology_prepare_intake.py
```

Check readiness:

```powershell
python scripts/cka_terminology_check_ready.py
```

Dry-run import:

```powershell
python scripts/cka_terminology_import_dry_run.py --terminology-root terminology_data
```

Run TERM-02 preflight gate:

```powershell
python scripts/cka_term02_preflight_gate.py
```

Run QA harness:

```powershell
python scripts/cka_terminology_run_qa.py
```

Inspect public reports:

```powershell
Get-ChildItem reports\cka_term*
```

## Required Before TERM-02

- TERM-01 through TERM-01H validations remain green.
- `LICENSE_ACK_PRIVATE.json` is present only locally.
- Preflight reports at least one import-ready system.
- Git staged paths exclude `terminology_data/` and `data/terminology/`.
- Public reports remain privacy-clean.

## Operator Commit Boundary

Commit source code and public reports only. Keep terminology files, private acknowledgments, generated databases, keys, PDFs, images, and archive files out of commits.
