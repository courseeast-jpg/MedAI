# MEDAI-CKA-TERM-INTEGRATION-PLAN-01 — Terminology and Coding Integration SPEC

Reports-only SPEC. Defines a safe, license-aware, privacy-safe plan
for the next terminology / coding integration phase **before** any
implementation begins. No runtime change. No tags created. PARK-20..23
tag pairs and the FREEZE tag pair remain intact. Cue expansion remains
explicitly **NOT** recommended.

## 1. Executive recommendation

After the v1 freeze (`7ef8ffd`), open the terminology / coding
integration track only through a reports-only SPEC (this block), and
let it gate the actual implementation block. The first safe
implementation block is **`MEDAI-CKA-TERM-INTEGRATION-NEXT-01`**, with
the strict scope and required-tests defined below. If any required
invariant cannot be met yet, open the safer intermediate
**`MEDAI-CKA-TERM-INTEGRATION-READINESS-02`** or
**`MEDAI-CKA-TERM-LICENSE-GATE-SPEC-02`** first.

This SPEC does not start implementation. It defines the rules,
license-class table, privacy gates, required tests, and rollback
conditions that any subsequent implementation block must satisfy.

## 2. Why this SPEC exists after ROADMAP-04

`ROADMAP-04` selected **FREEZE-MAINTENANCE-ONLY (G)** as the immediate
posture after the v1 freeze and named **`CKA-TERM-INTEGRATION-PLAN-01`**
as the SPEC that must open forward motion. Terminology / coding
integration is the highest-medical-value next lever but crosses
licensing and privacy boundaries that no single ROADMAP audit can
clear by itself. This block is the gate.

## 3. Current frozen release baseline

Pulled from `FREEZE-LOCAL-OPERATOR-RELEASE`, `ROADMAP-04`, and the
recent operator-readiness chain (public-safe metadata only):

| Signal | Value |
| --- | :-: |
| Branch | `clinical-knowledge-architecture` |
| HEAD before this block | `0dee189` (ROADMAP-04 receipt refresh) |
| Freeze commit | `7ef8ffd` |
| Both FREEZE tags resolve to freeze commit | true |
| `local_operator_release_frozen` | true |
| Cumulative `runtime_behavior_changed` across recent chain | false |
| `cue_expansion_recommended` | false |
| `external_api_used` / `external_api_enabled` | false / false |
| Whole MedAI project | **~94.0%** done / ~6.0% remaining |

## 4. Terminology / coding candidate inventory (from public-safe evidence)

From `reports/terminology_sources_preflight/` and the prior CKA-TERM
chain (CKA-TERM-01 through CKA-TERM-08, B07-TERM-01):

| Resource | Role | Already imported into private store? | Runtime integration enabled by default? |
| --- | --- | :-: | :-: |
| LOINC (`terminology_data/Loinc_2.82`) | primary | Yes — 109,325 rows per CKA-TERM-02 | No |
| RxNorm (`terminology_data/RxNorm_full_05042026`) | primary | Yes — 244,529 rows per CKA-TERM-02 | No |
| RxNorm prescribable subset | auxiliary | partial (preflight ready) | No |
| SNOMED CT US Edition | primary | preflight ready; rows not imported into runtime per `snomed_runtime_integration_enabled=false` | No |
| SNOMED CT International | secondary | preflight ready; not imported into runtime | No |
| UMLS Metathesaurus | separate future import | preflight ready; `umls_future_gated=true` | No |
| LICENSE_ACK_PRIVATE.json | private acknowledgement only | presence only | n/a |
| MedAI MKB-style internal coding | internal interface | n/a (internal) | already shipped, default-off |
| B07 terminology mapping interface | internal feature-flag layer | n/a (internal) | already shipped, default-off |

> Exact license terms are **not** claimed in any public-safe report
> reviewed. Every license-gated resource above is marked
> `requires_manual_license_verification_before_implementation: true`
> in the JSON.

## 5. License-class table

Six license classes are defined in the JSON
(`license_class_table.definitions`):

| Class | Meaning |
| --- | --- |
| `local_open_or_public_reference` | Freely redistributable; may be referenced by name in public reports. Rows still must not be embedded. |
| `local_license_gated_reference` | Exists locally under publisher license; presence may be referenced by canonical folder name only. Rows / license text never appear in public reports. |
| `local_private_acknowledgement_only` | Operator-supplied private license-acknowledgement file. Presence may be checked; contents never read or committed. |
| `prohibited_for_commit` | Must never be added to git. Enforced by `.gitignore`. |
| `runtime_private_store_only` | May be loaded into a private local DB/index for runtime lookup; that DB/index is itself `prohibited_for_commit`. |
| `aggregate_public_report_only` | May appear in public reports only as aggregate counts. |

Per-resource classifications live in
`license_class_table.entries` in the JSON. Summary:

- LOINC, RxNorm (+ prescribable), SNOMED CT US, SNOMED CT International,
  and UMLS → `local_license_gated_reference` +
  `runtime_private_store_only` + `prohibited_for_commit` +
  `aggregate_public_report_only`. All five also carry
  `requires_manual_license_verification_before_implementation: true`.
- `LICENSE_ACK_PRIVATE.json` →
  `local_private_acknowledgement_only` + `prohibited_for_commit`.
  Contents are never read in public reports.
- MedAI MKB-style internal coding and B07 mapping interface →
  `local_open_or_public_reference` + `aggregate_public_report_only`.

## 6. Privacy gates

- Public reports pass `clinical_knowledge.privacy.check_public_report_payload`
  before commit.
- Short SHAs (max ~7 chars) for any commit reference; full 40-char
  SHAs trigger the secret scanner.
- No `*.json` filename literals containing the word `report` in
  human-readable text; either rephrase or close the backtick before the
  extension to avoid the medical-filename pattern.
- Aggregate-only outputs (counts, distribution buckets, controlled-
  vocabulary tokens). No row content. No license text. No private
  filesystem paths.
- Anonymous identifiers (e.g. `record_001`) for any per-record
  reference in public reports.

## 7. Private-store and no-public-row-output rules

- **Private-store paths already `.gitignored`:** `terminology_data/`,
  `data/terminology/`, `config/terminology_sources.local.json`,
  `LICENSE_ACK_PRIVATE.json`, `**/LICENSE_ACK_PRIVATE*`,
  `**/*TERMINOLOGY_PRIVATE*`.
- **Access rules:** read-only access only; no public-report row dumps,
  only aggregate counts; no license text or license-acknowledgement
  contents ever written to public reports; no private paths printed in
  public reports (canonical folder names only); no external API calls
  to remote terminology services.
- **Prohibited outputs:** raw terminology rows in any committed file,
  license-acknowledgement contents in any committed file, private
  paths in public reports, PHI or source-document text in any
  committed file, secrets / keys / tokens in any committed file,
  clinical interpretation driven by terminology lookup, auto-acceptance
  driven by terminology match, DDI status changes driven by terminology
  coding, hypothesis promotion driven by terminology coding.

## 8. Future implementation architecture (11 components)

All eleven required components are defined in the JSON
(`future_integration_architecture`). Summary:

1. **Terminology source registry** — single read-only registry of
   canonical sources by name and license class.
2. **License-class gating layer** — refuse lookup on a source that
   still carries `requires_manual_license_verification: true`.
3. **Private local terminology store boundary** — local SQLite or
   sqlcipher index, `.gitignored`, read-only at runtime.
4. **No-public-row-output rule** — aggregate-only check before write.
5. **Aggregate-only reporting rule** — counts / distribution buckets /
   controlled-vocabulary fields only.
6. **Review-bound hypothesis annotation rule** — every coding
   annotation carries `review_required=true` and
   `auto_accept_allowed=false`, mirroring the DIAG-17/18 metadata
   pattern.
7. **Refusal for unsupported clinical inference** — terminology layer
   explicitly refuses to parse lab values, parse medication doses,
   change DDI status, infer diagnosis, infer treatment, or expand
   abbreviations. Every emitted plan publishes explicit
   `clinical_*_performed=false` flags.
8. **Local-only posture** — `MEDAI_LOCAL_ONLY=1` and
   `MEDAI_ALLOW_EXTERNAL_API=0` remain shipped defaults.
9. **No external API default** — external terminology APIs (UMLS REST,
   NLM web services, SNOMED Cloud) remain disabled unless an explicitly
   approved, separately-scoped block enables them with its own SPEC.
10. **Operator-visible wording boundaries** — controlled-vocabulary
    tokens (e.g. `terminology_match_hypothesis`) + a fixed disclaimer.
11. **Validation and rollback plan** — every new block in the track
    ships a focused pytest module asserting default-off behavior,
    aggregate-only output, and refusal of clinical inference. Rollback:
    unset the controlling feature flag; remove the runtime adapter
    import.

## 9. Required tests before implementation

19 focused tests must exist and pass before any merge in the next
block. The list lives in `required_tests_for_next_block` in the JSON.
Categories:

- Default-off behavior (unset / falsy / `os.environ` not polluted).
- Refusal of clinical inference (no recommendations, no DDI change,
  no hypothesis promotion, no lab/medication/abbreviation parsing).
- Privacy invariants (no licensed rows, no license text, no private
  paths in public reports; aggregate-only outputs).
- Frozen-artifact invariants (FREEZE + PARK-20..23 tag pairs still
  resolve; `app/main.py`, launchers, `app/startup_preflight.py`,
  `app/config.py` unchanged).

Deliverable: a focused pytest module under `tests/` named after the
chosen next block (e.g. `test_medai_cka_term_integration_next_01.py`).

## 10. First recommended implementation block

**`MEDAI-CKA-TERM-INTEGRATION-NEXT-01`** under the strict scope below:

- `default_off`: true
- `local_only`: true
- `no_licensed_rows_committed`: true
- `no_clinical_auto_accept`: true
- `no_diagnosis_or_treatment_inference`: true
- `no_ddi_behavior_change_unless_explicitly_scoped`: true
- `aggregate_only_public_reports`: true
- `review_bound_outputs`: true
- `tests_before_integration`: true

Scope sketch:

- Wire an env-gated, read-only terminology lookup helper that consults
  the existing private RxNorm / LOINC store already imported by
  `CKA-TERM-02` (244,529 + 109,325 rows = 353,854 concepts).
- Helper returns `None` when the feature flag
  `MEDAI_TERMINOLOGY_LOOKUP_ENABLED` is unset or falsy.
- When enabled and a record matches a positive signature, the helper
  emits a controlled-vocabulary `terminology_match_hypothesis`
  metadata dict carrying explicit `review_required=true` and
  `auto_accept_allowed=false` flags. **No row content from the private
  store appears in the dict.**
- No wiring into `app/main.py` runtime view in this block — the helper
  is a pure metadata function only. Streamlit wiring is a separate
  later block under its own approval.
- Focused pytest module covers every item in
  `required_tests_for_next_block`.

If any required invariant cannot be met yet, open the safer
intermediate **`MEDAI-CKA-TERM-INTEGRATION-READINESS-02`** (reports-
only readiness audit) or **`MEDAI-CKA-TERM-LICENSE-GATE-SPEC-02`**
(license-class-by-license-class verification block) first.

## 11. Explicit deferred items

The following are deferred and must not be reopened by the next
implementation block:

- **Wiring terminology output into `app/main.py` runtime** — deferred
  until a separate, approved `CKA-TERM-INTEGRATION-WIRING-NEXT-01`
  block.
- **UMLS lookup** — `umls_future_gated=true`; needs its own separately
  approved block.
- **SNOMED CT runtime integration** — `snomed_runtime_integration_enabled=false`;
  needs explicit enablement.
- **DDI logic changes** driven by terminology coding.
- **Diagnosis / treatment / dosing inference** driven by terminology.
- **Abbreviation expansion** in any form.
- **Cue-pack expansion** — explicitly **NOT** recommended across
  DIAG-13..21 and ROADMAP-01..04.
- **External terminology API enablement** — must remain off by
  default.
- **Public-report row dumps** under any condition.

## 12. Safety / privacy constraints

- No source documents opened.
- No raw OCR text, raw document text, raw filenames, private paths,
  PHI, secrets, runtime DB rows, backups, bundles, keys, or licensed
  terminology rows read.
- No `LICENSE_ACK_PRIVATE.json` contents read.
- No `app/main.py` / launcher / `app/startup_preflight.py` /
  `app/config.py` modification.
- No terminology imports performed.
- No tags created, moved, or deleted.
- No external APIs called or enabled.
- FREEZE tag pair (`7ef8ffd`) and PARK-20..23 tag pairs unchanged on
  origin.

## 13. Why cue expansion remains NOT recommended

Across DIAG-13..21 and ROADMAP-01..04 and the FREEZE block, cue
expansion has been ruled out as a primary lever. Adding cue packs
would reopen the classifier surface that the residual-Unknown track
explicitly stopped touching, expand the regression surface, create
new privacy / licensing responsibilities, and resolve no currently
failing operator outcome. This SPEC reaffirms:
`cue_expansion_recommended: false`.

## 14. Progress estimate

| Track | Before this SPEC | After this SPEC |
| --- | --- | --- |
| Whole MedAI project | ~94.0% done / ~6.0% remaining | ~94.1% done / ~5.9% remaining |

The SPEC produces no runtime change; the tiny bump reflects the
durable planning value of an explicit license-class table and
required-tests list before any implementation.
