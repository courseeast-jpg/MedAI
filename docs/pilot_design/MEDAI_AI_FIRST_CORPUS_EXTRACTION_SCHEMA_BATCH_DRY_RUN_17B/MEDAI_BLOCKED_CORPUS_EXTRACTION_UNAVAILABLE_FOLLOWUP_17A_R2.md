# MEDAI Blocked Corpus Extraction-Unavailable Follow-Up 17A-R2

17B is dry-run only. This follow-up records that the 587 blocked files are explicitly
excluded from AI extraction until fixed. This block does not repair them.

## Status

- Ready for AI extraction (17A): 12 files.
- Blocked for AI extraction (17A): 587 files.
- The blocked files carry `extraction_unavailable` (and related) blockers.

## Policy

- The 587 blocked files are excluded from AI extraction until their blockers are fixed
  in a separate, later block.
- No blocked file is uploaded, tokenized for outbound, or sent to any provider here.
- Temporary technical debt is accepted for UI and normalization work, not for privacy.
  Privacy gates are never deferred.

## Planned Follow-Up (Not Started)

A later 17A-R2 repair block would triage and remediate the `extraction_unavailable`
blockers (for example, unreadable formats, failed local extraction, or
normalization gaps) before any of those files could be considered for AI extraction.
That work is out of scope for 17B and is not started.
