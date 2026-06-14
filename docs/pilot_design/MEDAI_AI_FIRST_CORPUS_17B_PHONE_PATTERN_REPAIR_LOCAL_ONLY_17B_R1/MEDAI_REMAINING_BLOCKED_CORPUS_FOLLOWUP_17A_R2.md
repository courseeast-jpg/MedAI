# MEDAI Remaining Blocked Corpus Follow-Up 17A-R2

17B-R1 is local-only and does not repair the broader blocked corpus. This note records
that the 587 blocked files remain excluded from AI extraction until fixed.

## Status

- Ready for AI extraction: 12 files (the 17B-R1 repair target set).
- Blocked for AI extraction: 587 files, excluded until their blockers are fixed.

## Policy

- No blocked file is tokenized for outbound, uploaded, or sent to any provider here.
- Temporary technical debt is accepted for UI and normalization work, not for privacy.
  Privacy gates are never deferred.

## Planned Follow-Up (Not Started)

A later 17A-R2 repair block would triage and remediate the `extraction_unavailable`
blockers before any of those files could be considered for AI extraction. That work is
out of scope for 17B-R1 and is not started.
