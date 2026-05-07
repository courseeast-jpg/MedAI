# CKA-TERM-01B Import Planner Guide

TERM-01B prepares capacity planning only. It does not import terminology data.

## Dry-run command

```powershell
python scripts/cka_terminology_import_dry_run.py --json
```

Optional planning caps:

```powershell
python scripts/cka_terminology_import_dry_run.py --max-rows-per-file 100000 --chunk-size 5000 --json
```

## Boundaries

- Real import remains disabled by default.
- Synthetic test fixtures are allowed in temp directories only.
- Public summaries use counts, system labels, and safe IDs only.
- Actual import remains a TERM-02 task after licensed local files and private acknowledgement are provided.
