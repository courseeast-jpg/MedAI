# MEDAI-CKA-TERM-LICENSE-MANUAL-RETURN-PACK-04

## Executive Summary

This return pack gives the human operator a public-safe checklist for manual terminology license verification. It does not authorize implementation, real private-store access, terminology import, licensed row access, runtime DB access, external APIs, clinical inference, or cue expansion.

Private adapter implementation remains blocked until the operator completes license verification outside public reports and returns only safe confirmation status.

## Manual License Verification Checklist

The operator must verify, outside public reports, that each planned terminology resource may be used in the intended local private-store workflow.

Required confirmations:

- The resource license or terms allow the planned local use.
- The relevant license acknowledgement is available privately.
- The acknowledgement contents remain private.
- The terminology rows remain private.
- No row dumps, screenshots with licensed content, private paths, or DB exports are shared in public reports.
- No external API use is required.
- The future adapter remains read-only, fail-closed, and aggregate-only at the public boundary.

## Resource Verification Matrix

| Resource | Manual verification required | Private handling required | Public row output allowed | Implementation allowed now |
| --- | --- | --- | --- | --- |
| LOINC | Yes | Yes | No | No |
| RxNorm full | Yes | Yes | No | No |
| RxNorm prescribable | Yes | Yes | No | No |
| SNOMED CT US Edition | Yes | Yes | No | No |
| SNOMED CT International | Yes | Yes | No | No |
| UMLS Metathesaurus | Yes | Yes | No | No |
| LICENSE_ACK_PRIVATE.json | Yes | Yes | No | No |
| MedAI internal/public-safe MKB references | Yes, if used with terminology adapter | Review before public use | No private row output | No |
| B07 interface boundary | Yes, for boundary safety | Keep row content out | No | No |

## Private-Store Boundary Checklist

- Private store path/config is defined privately.
- Private config remains excluded from git.
- Terminology row contents remain excluded from public reports.
- Public layers receive aggregate metadata only.
- No runtime DB writes are allowed.
- No external API calls are allowed.
- Missing license state fails closed.
- Missing adapter fails closed.
- Unsafe output fields are rejected.
- No row codes, names, synonyms, definitions, concept rows, private paths, filenames, acknowledgement contents, PHI, or secrets cross into public reports.

## Allowed Return Artifacts

The operator may return only public-safe status artifacts:

- Completed checklist with yes/no/not-applicable status.
- Resource-level status with no license text and no row content.
- Confirmation that private evidence exists outside public reports.
- Confirmation that private config and store contents remain excluded from git.
- Confirmation that no external API use is required.

## Prohibited Return Artifacts

Prohibited artifacts:

- License acknowledgement contents.
- Licensed terminology rows.
- Row codes, names, synonyms, definitions, or concept rows.
- Screenshots containing licensed content.
- Private paths.
- Source filenames.
- DB exports.
- Runtime DB contents.
- Source/private documents.
- Raw OCR text or raw document text.
- Keys, secrets, PHI, or private corpus files.

## Authorization Gate

Before a private adapter implementation can be considered, all of the following must be true:

- Manual license verification is complete for each used resource.
- Private evidence remains private and is not committed.
- Private store/config boundary is defined.
- Fail-closed behavior is testable without exposing private contents.
- No-row-output rules are enforceable.
- Synthetic and fake-store tests are defined.
- Rollback and staged-safety checks are defined.

If any item is missing, only SPEC/readiness work may continue.

## Stop Conditions

Stop if license status is unclear, private evidence is missing, private content is exposed, row content appears in public output, DB writes are attempted, external APIs are attempted, clinical inference is introduced, DDI behavior changes, or cue expansion is attempted.

## Next Step

Recommended next block: `manual_operator_verification_required_before_more_implementation`.

Cue expansion remains NOT recommended.

## Progress Estimate

Whole MedAI done estimate: approximately 95.3%.
