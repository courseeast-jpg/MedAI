# B07-TERM Rollback Plan

The rollback plan is feature-flag first.

## Immediate Disable

Set these flags to their safe state:

- `MEDAI_B07_TERMINOLOGY_OPT_IN=false`
- `MEDAI_TERMINOLOGY_LOOKUP_ENABLED=false`
- `MEDAI_TERMINOLOGY_HYPOTHESIS_ANNOTATION=false`
- `MEDAI_TERMINOLOGY_READ_ONLY=true`
- `MEDAI_TERMINOLOGY_ALLOW_WRITES=false`

With these values, B07 must behave as it did before B07-TERM.

## Code Rollback

If a future implementation fails validation:

- revert only the B07-TERM implementation commit
- keep TERM-05 through TERM-08 artifacts intact
- rerun TERM-08, TERM-07, TERM-06, TERM-05, TERM-03, red-team, and final MVP validations
- confirm no private terminology artifact is staged

## Data Boundary

No rollback step may commit local terminology data, private acknowledgments, source terminology files, runtime DB/index files, keys, PDFs, images, or archives.

