# CKA-TERM-02 Stop-On-Failure Matrix

TERM-02 must stop immediately when any condition below is detected. The safe action is to fix the local setup or code scaffold, rerun validation, and continue only after the gate passes.

| Failure | Required Stop Reason | Safe Next Action |
| --- | --- | --- |
| Missing files | `no_supported_files_present` | Add licensed files locally, then rerun readiness. |
| Missing ack | `license_ack_private_missing` | Create the private acknowledgment locally, then rerun preflight. |
| License mismatch | `systems_pending_acknowledgment` | Update private acknowledgment to cover systems with files. |
| Staged terminology files | `terminology_files_staged` | Unstage private terminology files before continuing. |
| Terminology DB staged | `data_terminology_staged` | Unstage generated database files before continuing. |
| Row cap exceeded | `row_cap_exceeded` | Lower cap or split the run plan before import. |
| Malformed rows above threshold | `malformed_rows_above_threshold` | Stop and review parser safety before import. |
| Ambiguous lookup regression | `ambiguous_lookup_regression` | Restore ambiguity flagging before import. |
| Unknown code hallucination | `unknown_code_hallucination` | Stop; unknown terms must remain unmapped. |
| B07 promotes hypothesis | `b07_hypothesis_promotion` | Stop; B07 boundary must remain opt-in and non-promoting. |
| DDI status changed | `b07_ddi_status_changed` | Stop; local coding must not clear DDI state. |
| External API call | `external_api_attempted` | Stop; TERM-02 is local-only. |
| Privacy report leak | `privacy_report_leak` | Stop and remove raw private content from public reports. |
| Generated DB accidentally staged | `terminology_db_staged` | Unstage generated database files before commit. |

## Non-Negotiable Commit Boundary

- `terminology_data/` stays untracked.
- `data/terminology/` stays untracked.
- `LICENSE_ACK_PRIVATE.json` stays private.
- Database, key, private, PDF, image, and archive artifacts stay out of commits.
