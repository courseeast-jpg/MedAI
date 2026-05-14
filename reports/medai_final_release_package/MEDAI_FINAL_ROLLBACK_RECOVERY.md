# MedAI Final Rollback And Recovery Guide

## Restore From Git Tag

List release tags:

```powershell
git tag --list "medai-*2026-05-14"
```

Restore to the final release tag in a controlled worktree:

```powershell
git checkout medai-release-final-2026-05-14
```

For active development, create a new branch from the tag instead of editing the detached state.

## Restore From Git Bundle

Verify the bundle:

```powershell
git bundle verify backups/medai_final_release_2026-05-14.bundle
```

Clone from the bundle into a new location:

```powershell
git clone backups/medai_final_release_2026-05-14.bundle MedAI_release_restore
```

## Feature-Flag Rollback

Use these safe defaults to disable terminology-backed B07 effects:

```text
MEDAI_B07_TERMINOLOGY_OPT_IN=false
MEDAI_TERMINOLOGY_LOOKUP_ENABLED=false
MEDAI_TERMINOLOGY_HYPOTHESIS_ANNOTATION=false
MEDAI_TERMINOLOGY_READ_ONLY=true
MEDAI_TERMINOLOGY_ALLOW_WRITES=false
```

These settings preserve default-off, read-only, no-write terminology behavior.

## Route-Fix Rollback Considerations

ROUTE-FIX-01 changed fallback selection behavior in a narrow validated way. Any rollback should be a separate approval-gated block with focused routing tests and final MVP validation. Avoid mixing route rollback with terminology or clinical changes.

## Terminology Source Tooling Rollback Considerations

The source preflight and inventory tools are report-only. If rollback is needed, remove or disable those tools in a separate documentation/tooling block. Avoid touching private terminology source files or runtime stores.

## Validation Commands After Rollback

```powershell
python -m pytest tests
python scripts/run_cka_final_mvp_release_validation.py
python scripts/run_b07_term01_opt_in_integration_validation.py
python scripts/run_medai_route_fix01_validation.py
python scripts/run_medai_terminology_sources_preflight.py
python scripts/run_medai_terminology_inventory.py --terminology-root terminology_data
```
