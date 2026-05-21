# MEDAI-CKA-TERM-LICENSE-GATE-SPEC-02 Report

## Executive Recommendation

MedAI should add a readiness gate before any private terminology adapter work. The selected next block is `CKA-TERM-LICENSE-GATE-READINESS-03`.

Implementation is not recommended yet. This SPEC defines the gate, but it does not verify private license acknowledgement state, read licensed terminology rows, inspect private stores, or prove private adapter readiness.

## Why This SPEC Exists After PARK-01

`CKA-TERM-INTEGRATION-PARK-01` parked the default-off terminology helper and read-only Advanced technical details UI wiring. That mini-track proved safe synthetic metadata behavior, not real licensed terminology integration. The next boundary is license verification and private-store readiness.

## Current Parked Baseline

- Helper chain: complete
- UI wiring UAT: complete
- Frozen operator release preserved: true
- Freeze and terminology PARK tags untouched: true
- Runtime behavior changed by this SPEC: false
- Licensed terminology rows read: false
- LICENSE_ACK_PRIVATE.json read: false
- Terminology data staged: false
- External API used: false

## Resource License Gate Table

| Resource | License class | Private store only | Commit prohibited | Manual verification required | Public row output allowed | Public report mode | Runtime adapter before verification | External API allowed | Implementation allowed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LOINC | Licensed public terminology terms of use | Yes | Yes | Yes | No | Aggregate only | No | No | SPEC only |
| RxNorm full | Public federal terminology bundle with source vocabulary constraints | Yes | Yes | Yes | No | Aggregate only | No | No | SPEC only |
| RxNorm prescribable | Public federal subset with source vocabulary constraints | Yes | Yes | Yes | No | Aggregate only | No | No | SPEC only |
| SNOMED CT US Edition | Licensed controlled terminology | Yes | Yes | Yes | No | Aggregate only | No | No | SPEC only |
| SNOMED CT International | Licensed controlled terminology | Yes | Yes | Yes | No | Aggregate only | No | No | SPEC only |
| UMLS Metathesaurus | Licensed controlled terminology aggregation | Yes | Yes | Yes | No | Aggregate only | No | No | SPEC only |
| LICENSE_ACK_PRIVATE.json | Private local license acknowledgement | Yes | Yes | Yes | No | Not public | No | No | Readiness only |
| MedAI internal/public-safe MKB coding references | Public-safe internal reference if confirmed | No, after review | Review before commit | Yes | No private row output | Aggregate only | Only after review | No | SPEC/readiness |
| B07 mapping interface | Public-safe interface boundary if no licensed rows | No, after review | Review before commit | Yes | No private row output | Aggregate only | Only after review | No | SPEC/readiness |

## Private Adapter Preconditions

- Adapter must be injected.
- Adapter must be read-only by default.
- Adapter must fail closed on missing license acknowledgement.
- Adapter must fail closed on unsafe output fields.
- Adapter must never return licensed row content to public reports.
- Adapter must expose only controlled-vocabulary aggregate metadata.
- Adapter must not write DB state.
- Adapter must not call external APIs.
- Adapter must not require source documents, raw OCR text, raw document text, filenames, private paths, PHI, or secrets.
- Adapter outputs must remain review-bound and must not enable auto-accept.
- Adapter must not perform clinical interpretation, diagnosis inference, treatment inference, medication inference, lab value parsing, or DDI behavior.

## Public Report Restrictions

Public reports may contain aggregate counts and controlled status flags only. They must not contain row codes, display names, synonyms, definitions, raw concept rows, private paths, raw OCR text, raw document text, filenames, license acknowledgement contents, PHI, or secrets.

## Stop Conditions

Stop before implementation if license status is unclear, manual verification is missing, adapter output includes row content, public reports include row fields, private paths or secrets appear, a runtime DB write is attempted, an external API call is attempted, clinical inference is introduced, or cue expansion is attempted.

## Next Block Decision

Selected next block: `CKA-TERM-LICENSE-GATE-READINESS-03`.

`CKA-TERM-INTEGRATION-PRIVATE-ADAPTER-SPEC-01` remains deferred until readiness confirms license acknowledgement and private-store boundaries. `CKA-TERM-INTEGRATION-PRIVATE-ADAPTER-NEXT-01` is not recommended.

## Deferred Items

- Private adapter SPEC until readiness confirms license gates
- Private adapter implementation
- Real-corpus validation
- UMLS, SNOMED, DDI, diagnosis, and treatment work
- More Unknown diagnostics
- Cue expansion

Cue expansion remains NOT recommended.

## Safety And Privacy Constraints

This block is reports-only. It did not modify runtime code, `app/main.py`, helper code, Streamlit wiring, launcher files, startup preflight, config, OCR, extraction, classifier behavior, thresholds, cue packs, DDI behavior, or external API behavior. It did not open private documents, runtime DB rows, licensed terminology rows, private terminology stores, or license acknowledgement contents.

## Validation Results

| Check | Result |
| --- | --- |
| Public report privacy checks | Passed: 3/3 SPEC-02 reports privacy-clean |
| Final CKA MVP validation | Passed: 12/12 validation cases; 693 total tests passed; external API used false |
| B07 term01 validation | Passed: 6/6 cases; external API used false |
| ROUTE-FIX validation | Passed: medai_route_fix01_ready; external API used false |
| UI ops validation | Passed: medai_ui_ops_panel_ready |
| UI boot validation | Passed: medai_ui_boot_fix_startup_resilience_ready |
| Staged safety check | Passed: only SPEC-02 report files staged |

## Progress Estimate

Whole MedAI done estimate: approximately 94.9%.
