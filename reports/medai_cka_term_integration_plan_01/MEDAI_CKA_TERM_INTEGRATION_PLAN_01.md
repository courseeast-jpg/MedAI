# MEDAI-CKA-TERM-INTEGRATION-PLAN-01 — Short Summary

Reports-only SPEC for the next terminology / coding integration phase
after the v1 freeze. No implementation. No runtime change. No tags
created. FREEZE pair and PARK-20..23 pairs untouched. Cue expansion
remains explicitly **NOT** recommended.

## State

- Phase ID: `MEDAI-CKA-TERM-INTEGRATION-PLAN-01`
- Mode: `terminology_integration_spec`
- Branch: `clinical-knowledge-architecture`
- HEAD before this block: `0dee189` (ROADMAP-04 receipt refresh)
- Freeze commit: `7ef8ffd`
- `implementation_started`: **false**
- `runtime_behavior_changed`: **false**
- `cue_expansion_recommended`: **false**

## License-class table (summary)

| Resource | License class | Manual verification required |
| --- | --- | :-: |
| LOINC | license-gated + runtime-private-store-only + prohibited-for-commit + aggregate-public-report-only | **true** |
| RxNorm (+ prescribable) | same | **true** |
| SNOMED CT US | same | **true** |
| SNOMED CT International | same | **true** |
| UMLS | same + `umls_future_gated=true` | **true** |
| LICENSE_ACK_PRIVATE.json | private-acknowledgement-only + prohibited-for-commit | **true** |
| MedAI MKB internal coding | open-or-public-reference + aggregate-public-report-only | false |
| B07 mapping interface | open-or-public-reference + aggregate-public-report-only | false |

Exact license terms are **not** claimed; every gated resource carries
`requires_manual_license_verification_before_implementation: true`.

## Recommended next block

**`MEDAI-CKA-TERM-INTEGRATION-NEXT-01`** under the strict scope:

- `default_off`: true
- `local_only`: true
- `no_licensed_rows_committed`: true
- `no_clinical_auto_accept`: true
- `no_diagnosis_or_treatment_inference`: true
- `no_ddi_behavior_change_unless_explicitly_scoped`: true
- `aggregate_only_public_reports`: true
- `review_bound_outputs`: true
- `tests_before_integration`: true

Scope sketch: env-gated read-only terminology lookup helper consulting
the existing private RxNorm / LOINC store from `CKA-TERM-02` (353,854
concepts). Returns `None` when feature flag is unset / falsy. When
enabled and a record matches a positive signature, emits a
controlled-vocabulary `terminology_match_hypothesis` metadata dict with
`review_required=true` and `auto_accept_allowed=false`. No row content
in the dict. No Streamlit wiring. Focused pytest module covers all 19
required tests.

If any required invariant cannot be met yet: open the safer
intermediate **`MEDAI-CKA-TERM-INTEGRATION-READINESS-02`** or
**`MEDAI-CKA-TERM-LICENSE-GATE-SPEC-02`** first.

## Deferred items

- Wiring terminology output into `app/main.py` runtime.
- UMLS lookup (`umls_future_gated=true`).
- SNOMED CT runtime integration.
- DDI logic changes driven by terminology coding.
- Diagnosis / treatment / dosing inference driven by terminology.
- Abbreviation expansion.
- Cue-pack expansion (explicitly **NOT** recommended).
- External terminology API enablement.
- Public-report row dumps under any condition.

## Safety / privacy

- No source documents opened. No raw OCR text, raw document text, raw
  filenames, private paths, PHI, secrets, runtime DB rows, backups,
  bundles, keys, or licensed terminology rows read.
- No `LICENSE_ACK_PRIVATE.json` contents read.
- No `app/main.py` / launcher / preflight / config modification.
- No terminology imports performed.
- No tags created, moved, or deleted. FREEZE pair (`7ef8ffd`) and
  PARK-20..23 pairs untouched.
- No external APIs called or enabled.

## Progress

- Whole MedAI project: **~94.1%** done / ~5.9% remaining (small bump
  reflects durable planning value; no runtime change).
