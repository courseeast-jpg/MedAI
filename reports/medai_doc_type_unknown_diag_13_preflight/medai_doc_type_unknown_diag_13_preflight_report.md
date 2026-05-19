# MEDAI-DOC-TYPE-UNKNOWN-DIAG-13 Preflight — Remaining Lever Decision

This is a static, evaluation-only preflight. It ranks the remaining residual
Unknown pools and recommends the next safe lever for DIAG-13. It does not
implement anything and does not modify runtime behavior.

## Identity

- Block: `MEDAI-DOC-TYPE-UNKNOWN-DIAG-13-PREFLIGHT`
- Branch: `clinical-knowledge-architecture`
- HEAD commit (short): `aac7ff6`
- PARK-20 parking commit (short): `3e46461`

## Tag-status caveat

Branch is parked at PARK-20 (`3e46461`). PARK-20 tags
(`medai-unknown-diag-language-metadata-ready-2026-05-19` and
`medai-final-parked-post-unknown-diag-language-metadata-2026-05-19`) exist
locally and correctly target `3e46461`, but the remote tag push is currently
blocked by HTTP 403 on `origin`'s receive-pack endpoint. The out-of-band
GitHub/proxy permission fix remains pending. This preflight block does not
touch tags.

## Completed language-detector metadata tracks

| Track | Blocks | Env flag | Slice | Default off | Review-bound | Auto-accept | Clinical behavior | Parsing | External API | Data-layer Unknown reduction |
| --- | --- | --- | ---: | :-: | :-: | :-: | :-: | :-: | :-: | ---: |
| Numeric-table safe-default | DIAG-06A / 07A / 08A | `MEDAI_DOC_TYPE_OPERATOR_REVIEW_BADGE_ENABLED` | 11 | yes | yes | no | no | no | no | 0 |
| Language-propagation metadata | DIAG-09A / 10A | `MEDAI_DOC_TYPE_LANGUAGE_PROPAGATION_METADATA_ENABLED` | 11 | yes | yes | no | no | no | no | 0 |
| Latin abbreviation metadata | DIAG-11A / 12A | `MEDAI_DOC_TYPE_LATIN_ABBREVIATION_METADATA_ENABLED` | 8 | yes | yes | no | no | no | no | 0 |

All three are review-only, env-gated, and never modify the data-layer
document type.

## Remaining deferred pools

| Pool | Records |
| --- | ---: |
| Table-header special case | 1 |
| Text-layer | 21 |
| `fallback_ran_but_no_family_match` | 17 |
| `ambiguous_below_threshold` | 15 |

## Ranking criteria

Each candidate lever is ranked by:

- safety risk
- expected operational value
- false-positive risk
- likelihood of privacy-safe aggregate analysis
- whether it requires runtime behavior
- whether it can remain evaluation-only

## Ranked candidate levers

### Rank 1 — Candidate B: text-layer diagnostic (21 records)

- Safety risk: low
- Expected operational value: high
- False-positive risk: low
- Likelihood of privacy-safe aggregate analysis: high
- Requires runtime behavior: no
- Can remain evaluation-only: yes

**Rationale.** Largest residual pool. Mirrors the diagnostic shape of
DIAG-02..05 that already shipped privacy-safe. A DIAG-13 block here can be
purely aggregate: characterize how text-layer extraction interacts with
classifier outcomes (counts, structural shapes, cue-coverage statistics)
without touching OCR routing, OCR engine behavior, classifier behavior,
thresholds, or cue packs. No operator surface or default-off helper is
required at the diagnostic step.

**Recommendation:** preferred first choice.

### Rank 2 — Candidate C: fallback shape audit (17 records)

- Safety risk: low
- Expected operational value: medium
- False-positive risk: low to medium
- Likelihood of privacy-safe aggregate analysis: high
- Requires runtime behavior: no
- Can remain evaluation-only: yes

**Rationale.** Second-largest pool. An aggregate audit of
`fallback_ran_but_no_family_match` shapes (counts, cue-coverage gaps,
structural patterns) is safe so long as it stays diagnostic. Cue expansion
remains not recommended; this audit only characterizes the gap, it does not
close it. Best taken after the text-layer diagnostic to avoid premature cue
work.

**Recommendation:** second choice.

### Rank 3 — Candidate A: table-header special case (1 record)

- Safety risk: low
- Expected operational value: low
- False-positive risk: low
- Likelihood of privacy-safe aggregate analysis: high but sample size is
  minimal
- Requires runtime behavior: no
- Can remain evaluation-only: yes

**Rationale.** Pool size of 1 is too small to justify a dedicated DIAG block
unless this single record is blocking an operational flow. It is not
currently described as blocking. Deferred; can be rolled into a later
consolidated cleanup pass.

**Recommendation:** defer unless it blocks operational flow.

### Rank 4 — Candidate D: ambiguous below threshold (15 records)

- Safety risk: medium
- Expected operational value: medium
- False-positive risk: higher
- Likelihood of privacy-safe aggregate analysis: medium
- Requires runtime behavior: no
- Can remain evaluation-only: yes

**Rationale.** Below-threshold ambiguity has the highest false-positive
risk among the four pools because action on it tends to drift toward
threshold lowering, cue expansion, or scoring changes — all of which are
explicitly out of scope for the current safety posture. If addressed at
all, must remain strictly aggregate-only and must not propose threshold or
scoring changes.

**Recommendation:** defer to last; avoid threshold changes.

## Recommended next lever

| Position | Candidate | Records |
| --- | --- | ---: |
| Primary | B — text-layer diagnostic | 21 |
| Secondary | C — fallback shape audit | 17 |
| Deferred (low value) | A — table-header special case | 1 |
| Deferred (higher risk) | D — ambiguous below threshold | 15 |

Constraints for whatever lever is picked:

- Must remain evaluation-only.
- Must remain aggregate-only at the diagnostic stage.
- Must not propose cue expansion.
- Must not propose threshold or scoring changes.

## Block flags

- `behavior_changed`: false
- `external_api_used`: false
- `cue_expansion_recommended`: false
- `implementation_started`: false
- `ocr_routing_changed`: false
- `ocr_engine_behavior_changed`: false
- `classifier_behavior_changed`: false
- `thresholds_or_scoring_changed`: false
- `cue_packs_added`: false
- `lab_values_parsed`: false
- `medications_or_ddi_parsed`: false
- `abbreviations_parsed_or_expanded`: false
- `clinical_interpretation_added`: false
- `external_api_enabled`: false
- `diag_13_implementation_started`: false
- `tag_push_work_started_in_this_block`: false

## Safety / privacy statement

This block is a static, evaluation-only preflight that ranks four
already-known residual pools using counts and qualitative criteria. No
source documents, raw OCR text, raw document text, raw filenames, private
paths, PHI, secrets, DBs, backups, or bundles are referenced. No runtime
code is modified. No DIAG-13 implementation begins.

## Progress estimate

| Track | Before | After (if preflight completes) |
| --- | --- | --- |
| Residual Unknown-reduction | ~96% done / ~4% remaining | ~97% done / ~3% remaining |
| Whole MedAI project | ~86% done / ~14% remaining | ~86% done / ~14% remaining |
