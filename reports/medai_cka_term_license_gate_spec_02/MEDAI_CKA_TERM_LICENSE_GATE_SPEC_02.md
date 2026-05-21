# MEDAI-CKA-TERM-LICENSE-GATE-SPEC-02

## Executive Recommendation

Create a strict license and private-store readiness gate before any real RxNorm, LOINC, SNOMED CT, UMLS, or other licensed terminology resource is connected to MedAI runtime workflows.

The next block should be `CKA-TERM-LICENSE-GATE-READINESS-03`, not implementation. Manual license verification and private-store readiness must be proven before a private terminology adapter is designed or implemented against real licensed resources.

## Why This SPEC Exists

`CKA-TERM-INTEGRATION-PARK-01` froze the first terminology helper and UI wiring mini-track. That chain is default-off, read-only, synthetic-only, review-bound, and aggregate-only. The next risk boundary is licensing and private-store access. This SPEC defines the gate that must exist before using real licensed terminology resources.

## Current Parked Baseline

- Terminology helper chain: complete
- Terminology UI wiring UAT: complete
- Local operator release: frozen
- Runtime/default behavior changed by this SPEC: false
- Licensed terminology rows read: false
- LICENSE_ACK_PRIVATE.json read: false
- Terminology data staged: false
- External API used: false
- Cue expansion recommended: false

## Resource License Gate Table

| Resource | License class | Private store only | Prohibited for commit | Manual verification required | Public row output allowed | Aggregate public report only | Runtime adapter before verification | External API allowed | Implementation block allowed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LOINC | Licensed public terminology with terms of use | Yes | Yes | Yes | No | Yes | No | No | SPEC only |
| RxNorm full | Licensed/public federal terminology bundle; source vocabularies may carry constraints | Yes | Yes | Yes | No | Yes | No | No | SPEC only |
| RxNorm prescribable | Licensed/public federal terminology subset; source constraints still require review | Yes | Yes | Yes | No | Yes | No | No | SPEC only |
| SNOMED CT US Edition | Licensed controlled terminology | Yes | Yes | Yes | No | Yes | No | No | SPEC only |
| SNOMED CT International | Licensed controlled terminology | Yes | Yes | Yes | No | Yes | No | No | SPEC only |
| UMLS Metathesaurus | Licensed controlled terminology aggregation | Yes | Yes | Yes | No | Yes | No | No | SPEC only |
| LICENSE_ACK_PRIVATE.json | Private local license acknowledgement | Yes | Yes | Yes | No | No | No | No | Readiness only |
| MedAI internal/public-safe MKB coding references | Public-safe internal references only | No, if confirmed public-safe | Review before commit | Yes, for provenance | No row-level private output | Yes | Only after review | No | SPEC/readiness |
| B07 mapping interface | Public-safe interface boundary | No, if no licensed rows | Review before commit | Yes | No | Yes | Only after review | No | SPEC/readiness |

## Private Adapter Preconditions

- Adapter must be injected, not globally imported from private stores.
- Adapter must be read-only by default.
- Adapter must fail closed when license acknowledgement is missing or cannot be verified.
- Adapter must fail closed when unsafe fields are returned.
- Adapter must never return licensed row content to public reports.
- Adapter must expose only controlled-vocabulary aggregate metadata.
- Adapter must not write DB state.
- Adapter must not call external APIs.
- Adapter must not require source documents, raw OCR text, raw document text, filenames, private paths, PHI, or secrets.
- Adapter output must remain review-bound and must never enable auto-accept.
- Adapter must not perform clinical interpretation, diagnosis inference, treatment inference, medication inference, lab value parsing, or DDI behavior.

## Public Report Restrictions

Public reports must contain aggregate counts and controlled status flags only. They must not include:

- Row codes
- Display names
- Synonyms
- Definitions
- Raw concept rows
- Private paths
- Raw OCR text
- Raw document text
- Filenames
- License acknowledgement contents
- PHI
- Secrets

## Stop Conditions

Stop before implementation if any of these occur:

- License status is unclear.
- Manual license verification is missing.
- Adapter returns row content.
- Public reports contain row fields.
- Private paths or secrets are detected.
- Runtime DB write is attempted.
- External API is attempted.
- Clinical inference is introduced.
- Cue expansion is attempted.

## Next Block Decision

Selected next block: `CKA-TERM-LICENSE-GATE-READINESS-03`.

Reason: this SPEC defines the required gate, but it does not verify actual license acknowledgement state and intentionally does not read private license files or terminology rows. A readiness audit should prove manual verification and private-store boundaries before any private adapter SPEC or implementation.

## Deferred Items

- `CKA-TERM-INTEGRATION-PRIVATE-ADAPTER-SPEC-01` is deferred until readiness confirms license gates.
- `CKA-TERM-INTEGRATION-PRIVATE-ADAPTER-NEXT-01` is not recommended.
- Real-corpus validation is deferred.
- UMLS/SNOMED/DDI/diagnosis/treatment work is deferred.
- Cue expansion remains NOT recommended.

## Progress Estimate

Whole MedAI done estimate: approximately 94.9%.

