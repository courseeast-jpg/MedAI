# MEDAI-DB-REPAIR-01 MKB DB Diagnostics

This block diagnoses local MKB startup database state without printing rows or key values.

The quarantine/recreate workflow is confirmation-gated and never runs automatically.

Suggested operator sequence:

1. Review the diagnostics report.
2. Approve quarantine/recreate only when replacing the local startup DB with an empty schema DB is acceptable.
3. Launch the UI after the approved repair step.
