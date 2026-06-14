# 17C-R2 Public Report Redaction Policy (R9)

## Problem
The real 17C-R2 live run wrote private filesystem paths and a secret-like digest into the
committed public reports. The privacy checker flagged `summary.json`:
- `private_filename_path_leaks = 3` (three `WIN_PATH` values)
- `secret_leaks = 1` (a 64-hex digest classified as `SECRET`)

`test_public_reports_carry_no_phi_or_secret` therefore failed. The fix repairs the report
content — it does **not** weaken the test.

## Redaction labels (status-preserving, no raw value)
| Label | Replaces |
| --- | --- |
| `PRIVATE_STAGING_PATH_REDACTED` | live request/response staging paths |
| `PRIVATE_CHECKPOINT_PATH_REDACTED` | durable checkpoint path |
| `PRIVATE_EVIDENCE_PATH_REDACTED` | preserved failure-evidence path |
| `PRIVATE_PATH_REDACTED` | any other private filesystem path |
| `SECRET_REDACTED` | secret-like token / 64-hex digest |

## Helper
`execution/public_report_redaction.py` is driven by the **same** detector the privacy
checker uses (`clinical_knowledge.privacy.sanitizer.sanitize_text`). Anything flagged as a
private reference (`WIN_PATH` / `UNIX_PATH` / `MEDICAL_FILENAME`) or a secret
(`ALWAYS_BLOCK` / `SECRET`) is replaced with the appropriate label:
- JSON reports are redacted structurally (output stays valid JSON).
- Markdown/CSV are redacted by detector offsets.

## Result
- `summary.json`: 3 path + 1 secret redactions applied → 0 leaks after.
- All 10 scanned 17C-R2 live-batch public reports pass the privacy checker after redaction.
- `evidence_preservation_path` debugging value is preserved as the label
  `PRIVATE_EVIDENCE_PATH_REDACTED` plus `evidence_preserved=true`; no raw Windows path
  remains in any public report.

## Invariants
The privacy checker and its tests are not weakened. Real private locations remain only in
private files outside the repo. No raw response body, tokenized payload, token map,
private identifier, or credential is committed.
