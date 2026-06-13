# Vertex Semantic Package — Operator Review Drafts (15Q, no-live replay)

Replayed from recorded 15P-C / 15P-D live comparison reports. No provider call is made here.
All drafts are review-bound: active MKB writes = 0, auto-accept = off.

## Cytology/pathology narrative package (`cytology_pathology_narrative`)

- Provider route: `vertex` | Model: `gemini-2.5-flash-lite`
- Replayed from recorded 15P-C/15P-D report; no live call in this view
- Review required: `True` | Auto-accept: off | Active MKB writes: 0
- Hallucinated field count: `0` | Schema valid: `True`

### Source visible body

> Synthetic narrative fixture with tests ordered, diagnosis, and recommendation sections.

### Source sections

- Tests Ordered: tests ordered section present
- Diagnosis: diagnosis section present
- Recommendation: recommendation section present

### Vertex semantic findings (separated from source body)

| Label | Value | Section | Evidence text | Uncertainty | Unknown |
| --- | --- | --- | --- | --- | --- |
| Tests Ordered | source-visible narrative present | Tests Ordered | tests ordered section present | source-visible candidate fact | False |
| Diagnosis | source-visible narrative present | Diagnosis | diagnosis section present | source-visible candidate fact | False |
| Recommendation | source-visible narrative present | Recommendation | recommendation section present | source-visible candidate fact | False |

### Evidence anchors

- `cyto_a1` [Tests Ordered]: tests ordered section present
- `cyto_a2` [Diagnosis]: diagnosis section present
- `cyto_a3` [Recommendation]: recommendation section present

### Unknown / missing values

- (none unknown in this synthetic family)

### Uncertainty flags

- Tests Ordered: source-visible candidate fact; operator must compare
- Diagnosis: source-visible candidate fact; operator must compare
- Recommendation: source-visible candidate fact; operator must compare

### Operator controls (simulation only — no active MKB write)

- [ ] Accept for review queue only
- [ ] Reject
- [ ] Defer

## Urinalysis/table-like lab package (`urinalysis_table_like_lab`)

- Provider route: `vertex` | Model: `gemini-2.5-flash-lite`
- Replayed from recorded 15P-C/15P-D report; no live call in this view
- Review required: `True` | Auto-accept: off | Active MKB writes: 0
- Hallucinated field count: `0` | Schema valid: `True`

### Source visible body

> Synthetic table fixture with specific gravity, pH, occult blood, and RBC rows.

### Source sections

- Urinalysis Table: table rows grouped under urinalysis table

### Vertex semantic findings (separated from source body)

| Label | Value | Section | Evidence text | Uncertainty | Unknown |
| --- | --- | --- | --- | --- | --- |
| Specific Gravity | 1.020 | Urinalysis Table | table rows grouped under urinalysis table | source-visible candidate fact | False |
| pH | 7.5 | Urinalysis Table | table rows grouped under urinalysis table | source-visible candidate fact | False |
| Occult Blood | Trace | Urinalysis Table | table rows grouped under urinalysis table | source-visible candidate fact | False |
| RBC | 3-10 | Urinalysis Table | table rows grouped under urinalysis table | source-visible candidate fact | False |

### Evidence anchors

- `ua_a1` [Urinalysis Table]: table rows grouped under urinalysis table

### Unknown / missing values

- (none unknown in this synthetic family)

### Uncertainty flags

- Specific Gravity: source-visible candidate fact; operator must compare
- pH: source-visible candidate fact; operator must compare
- Occult Blood: source-visible candidate fact; operator must compare
- RBC: source-visible candidate fact; operator must compare

### Operator controls (simulation only — no active MKB write)

- [ ] Accept for review queue only
- [ ] Reject
- [ ] Defer

## Portal result-card package (`portal_result_cards`)

- Provider route: `vertex` | Model: `gemini-2.5-flash-lite`
- Replayed from recorded 15P-C/15P-D report; no live call in this view
- Review required: `True` | Auto-accept: off | Active MKB writes: 0
- Hallucinated field count: `0` | Schema valid: `True`

### Source visible body

> Synthetic portal cards fixture with card labels and visible reference text.

### Source sections

- Portal Result Cards: card labels grouped as portal result cards

### Vertex semantic findings (separated from source body)

| Label | Value | Section | Evidence text | Uncertainty | Unknown |
| --- | --- | --- | --- | --- | --- |
| Urine Color | Orange | Portal Result Cards | card labels grouped as portal result cards | source-visible candidate fact | False |
| Appearance | Clear | Portal Result Cards | card labels grouped as portal result cards | source-visible candidate fact | False |
| Leukocyte Esterase | Negative | Portal Result Cards | card labels grouped as portal result cards | source-visible candidate fact | False |
| Protein | Trace | Portal Result Cards | card labels grouped as portal result cards | source-visible candidate fact | False |

### Evidence anchors

- `portal_a1` [Portal Result Cards]: card labels grouped as portal result cards

### Unknown / missing values

- (none unknown in this synthetic family)

### Uncertainty flags

- Urine Color: source-visible candidate fact; operator must compare
- Appearance: source-visible candidate fact; operator must compare
- Leukocyte Esterase: source-visible candidate fact; operator must compare
- Protein: source-visible candidate fact; operator must compare

### Operator controls (simulation only — no active MKB write)

- [ ] Accept for review queue only
- [ ] Reject
- [ ] Defer

## Mixed narrative + numeric result package (`mixed_narrative_numeric_result`)

- Provider route: `vertex` | Model: `gemini-2.5-flash-lite`
- Replayed from recorded 15P-C/15P-D report; no live call in this view
- Review required: `True` | Auto-accept: off | Active MKB writes: 0
- Hallucinated field count: `0` | Schema valid: `True`

### Source visible body

> Synthetic mixed fixture with a clinical note, visible culture result, unknown collection time, and nitrite result.

### Source sections

- Clinical Note: narrative section present
- Result Summary: numeric/result section present

### Vertex semantic findings (separated from source body)

| Label | Value | Section | Evidence text | Uncertainty | Unknown |
| --- | --- | --- | --- | --- | --- |
| Clinical Note | source-visible narrative present | Clinical Note | narrative section present | source-visible candidate fact | False |
| Culture Result | No growth | Result Summary | numeric/result section present | source-visible candidate fact | False |
| Collection Time | Unknown | Result Summary | numeric/result section present | missing or not visible in source | True |
| Nitrite | Negative | Result Summary | numeric/result section present | source-visible candidate fact | False |

### Evidence anchors

- `mixed_a1` [Clinical Note]: narrative section present
- `mixed_a2` [Result Summary]: numeric/result section present

### Unknown / missing values

- Collection Time

### Uncertainty flags

- Clinical Note: source-visible candidate fact; operator must compare
- Culture Result: source-visible candidate fact; operator must compare
- Collection Time: missing or not visible in source
- Nitrite: source-visible candidate fact; operator must compare

### Operator controls (simulation only — no active MKB write)

- [ ] Accept for review queue only
- [ ] Reject
- [ ] Defer
