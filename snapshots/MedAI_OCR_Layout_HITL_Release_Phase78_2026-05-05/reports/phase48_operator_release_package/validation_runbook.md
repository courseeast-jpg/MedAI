# Validation Runbook

Run these commands from `G:\Codex\2026-04-22-connect-github`.

```powershell
git status --short
git rev-parse HEAD
python -m pytest tests
python scripts\run_phase46_validation_drift_lock.py
python scripts\run_phase47_final_regression_hardening.py
```

Optional extended validation:

```powershell
python scripts\run_phase44_cyrillic_ocr_validation.py
python scripts\run_phase45_cyrillic_nonlab_review_validation.py
```

Expected outputs:

- Tests pass.
- Phase46 conclusion is `deterministic_lock_ready`.
- Phase47 conclusion is `release_candidate_ready`.
- Safety regression is `False`.
- PHI/report artifacts tracked is `False`.
- Count convention remains `total == accepted + review`, with `review_ocr_quality` and `empty` as review subsets.

If any safety regression, unexpected accepted increase, tracked report PDF, or unclassified runtime drift appears, stop release use and investigate before processing more documents.
