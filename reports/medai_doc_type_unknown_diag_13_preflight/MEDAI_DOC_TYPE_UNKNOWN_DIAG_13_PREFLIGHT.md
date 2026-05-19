# MEDAI-DOC-TYPE-UNKNOWN-DIAG-13 Preflight

Static, evaluation-only preflight that ranks the next safe Unknown-reduction
lever for DIAG-13. No implementation. No runtime behavior change.

## State

- Branch: `clinical-knowledge-architecture`
- HEAD commit (short): `aac7ff6`
- PARK-20 parking commit (short): `3e46461`

## Tag status caveat

Branch parked at PARK-20 (`3e46461`). Local PARK-20 tags target `3e46461`
correctly; remote tag push remains blocked by HTTP 403 pending an
out-of-band GitHub/proxy permission fix. This preflight block does not
touch tags.

## Remaining deferred pools

| Pool | Records |
| --- | ---: |
| Table-header special case | 1 |
| Text-layer | 21 |
| `fallback_ran_but_no_family_match` | 17 |
| `ambiguous_below_threshold` | 15 |

## Ranked next-lever recommendation

| Rank | Candidate | Pool | Records | Note |
| ---: | --- | --- | ---: | --- |
| 1 | B | text-layer diagnostic | 21 | Preferred; aggregate-only; mirrors DIAG-02..05 pattern. |
| 2 | C | fallback shape audit | 17 | Second; aggregate-only; no cue expansion. |
| 3 | A | table-header special case | 1 | Defer; sample too small. |
| 4 | D | ambiguous below threshold | 15 | Defer; highest false-positive risk; avoid threshold changes. |

Constraints if any lever is picked next:

- Evaluation-only.
- Aggregate-only at the diagnostic stage.
- No cue expansion.
- No threshold or scoring changes.

## Flags

- `behavior_changed`: false
- `external_api_used`: false
- `cue_expansion_recommended`: false
- `implementation_started`: false

## Safety / privacy

Static text only. No source documents, raw OCR text, raw document text,
raw filenames, private paths, PHI, secrets, DBs, backups, or bundles are
referenced.

## Progress estimate

- Before: residual Unknown ~96% done / ~4% remaining; whole project ~86%
  done / ~14% remaining.
- After (if preflight completes): residual Unknown ~97% done / ~3%
  remaining; whole project ~86% done / ~14% remaining.
