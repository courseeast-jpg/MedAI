# MEDAI Remaining 587 Blocked Files Not Included In 17C

17C processes exactly the 12 validated tokenized requests from 17B-R1. The 587 blocked
files are explicitly excluded and are not uploaded, tokenized for outbound, or sent to
any provider in 17C.

- Ready (17C scope): 12 validated tokenized requests.
- Blocked (excluded): 587 files with `extraction_unavailable` and related blockers.

The blocked files remain excluded until a separate, later repair block fixes their
blockers. Temporary technical debt is accepted for UI and normalization work, not for
privacy. Privacy gates are never deferred.
