# CKA-TERM-01D Terminology QA Guide

TERM-01D provides a synthetic golden lookup harness that TERM-02 can reuse after real local imports are available.

## Command

```powershell
python scripts/cka_terminology_run_qa.py --json
```

## Boundaries

- Uses synthetic fixtures only.
- Unknown terms must remain unmapped.
- Ambiguous terms must be flagged.
- B07 local terminology lookup remains opt-in.
- No external APIs are used.
