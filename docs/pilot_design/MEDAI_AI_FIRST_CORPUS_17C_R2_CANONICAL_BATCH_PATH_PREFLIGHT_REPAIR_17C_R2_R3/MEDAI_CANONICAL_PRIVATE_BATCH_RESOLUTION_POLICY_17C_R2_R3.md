# MEDAI Canonical Private Batch Resolution Policy 17C-R2-R3

## Resolver Order

`execution/canonical_batch_paths.resolve_canonical_batch()` returns the first existing
candidate among:

1. The explicit expected path
   `C:\Users\S1\AppData\Local\MedAI_Private\ai_extraction_17B_R2_R1_478_repaired\outbound_requests_private.jsonl`.
2. `%LOCALAPPDATA%\MedAI_Private\ai_extraction_17B_R2_R1_478_repaired\outbound_requests_private.jsonl`
   (only when `LOCALAPPDATA` is set).
3. `Path.home()/AppData/Local/MedAI_Private/ai_extraction_17B_R2_R1_478_repaired/outbound_requests_private.jsonl`.

If none exists, the explicit expected path is returned with an exists=False flag for
reporting. A read-only file resolves normally (read-only is readable; never "missing").

## Rules

- The resolver must not be used to weaken any validation; it only locates the file.
- Callers still verify SHA256 against the sidecar, the 478 physical-newline record
  count, 0 malformed, 0 residual PI, and credentials before any future live run.
- Private paths/values are never printed or committed; only the resolved path string
  (a directory/file path, not contents) appears in reports.
