# MEDAI-CKA-TERM-INTEGRATION-PRIVATE-ADAPTER-SPEC-01

## Executive Recommendation

Create a design-only future private terminology adapter boundary. The selected next block is `CKA-TERM-INTEGRATION-PRIVATE-ADAPTER-READINESS-02`.

Private adapter implementation remains disallowed because manual license verification and private-store operational details remain open.

## Why This SPEC Exists After READINESS-03

`CKA-TERM-LICENSE-GATE-READINESS-03` found the gate ready for a reports-only private adapter SPEC. It did not authorize implementation, licensed row access, terminology import, private store reads, runtime DB writes, external APIs, or clinical inference.

## Current Parked Baseline

- Terminology helper and UI wiring mini-track: parked
- License gate readiness: ready for private adapter SPEC
- Runtime/default behavior changed by this SPEC: false
- Private adapter implemented: false
- Licensed terminology rows read: false
- Private store opened or read: false
- LICENSE_ACK_PRIVATE.json read: false
- Cue expansion recommended: false

## Adapter Contract

The future adapter is a read-only lookup boundary for private local terminology stores. It must be injected by the caller, disabled by default, local-only, and unable to call external APIs. It must never return licensed row content to public reports.

Allowed adapter role:

- Accept controlled candidate lookup requests.
- Check license-state readiness through a private gate without exposing acknowledgement contents.
- Return aggregate controlled-vocabulary metadata only.
- Fail closed on any unsafe state.

Disallowed adapter role:

- No row rendering.
- No terminology import.
- No DB writes.
- No external API calls.
- No clinical interpretation.
- No diagnosis, treatment, medication, dose, lab value, or DDI inference.
- No cue expansion.

## Input Contract

Allowed inputs are controlled candidate queries only. The adapter must reject or ignore raw/private inputs.

Disallowed inputs:

- Raw OCR text
- Raw document text
- Source filenames
- Private paths
- PHI
- Secrets
- Free-text clinical interpretation
- Runtime DB rows
- Source/private documents

## Output Contract

Allowed output fields:

- `match_family`
- `terminology_system_family`
- `matches_count`
- `license_class`
- `review_required`
- `auto_accept_allowed`
- `public_report_safe`

Required output semantics:

- `review_required` must be true.
- `auto_accept_allowed` must be false.
- `public_report_safe` must be true before output crosses into a public report or UI layer.

## Disallowed Output Fields

- Row code
- Display name
- Synonym
- Definition
- Raw concept row
- Private path
- License acknowledgement content
- Source filename
- Raw OCR text
- Raw document text
- PHI
- Secrets

## License-State Gate

Manual license verification is required before implementation. A future private gate may presence-check `LICENSE_ACK_PRIVATE.json`, but acknowledgement contents must never enter public reports, logs, or UI output.

Missing, unclear, or unverifiable license state must fail closed and return no metadata.

## Fail-Closed Behavior

- Missing adapter: no output.
- Missing license state: no output.
- Unsafe output fields: reject output.
- Adapter exception: no output plus safe diagnostic only.
- Private store missing: no output.
- License state unclear: no output.
- Public safety flag false: no output.

## Private Store Boundary

The future private store must be local-only, read-only by default, and configured through local private configuration. Public reports may only receive aggregate counts and controlled status fields. No row content, private path, filename, or acknowledgement content may cross the boundary.

## Test Plan

- Synthetic-only unit tests.
- Fake private-store adapter contract tests.
- Default-off tests.
- Missing license state fails-closed tests.
- Missing adapter fails-closed tests.
- Unsafe row content rejection tests.
- Raw/private input rejection tests.
- No-row-output tests.
- Public report privacy checks.
- No runtime DB write tests.
- No external API tests.
- No clinical inference tests.
- No DDI behavior change tests.
- Cue expansion remains false tests.

## Implementation Prerequisites

- Manual license verification complete.
- Private store path configured through local-only config.
- License-state gate designed and tested.
- Adapter injection boundary defined.
- No row content returned to public layer.
- Rollback plan defined.
- Staged safety gate defined.
- Public-report privacy checker extended if needed.

## Stop Conditions

Stop before implementation if license verification is incomplete, private store contract is unclear, adapter returns row content, public reports contain row fields, private paths or secrets appear, DB writes are attempted, external APIs are attempted, clinical inference is introduced, DDI behavior changes, or cue expansion is attempted.

## Next Block Decision

Selected next block: `CKA-TERM-INTEGRATION-PRIVATE-ADAPTER-READINESS-02`.

Implementation is not allowed.

## Deferred Items

- Private adapter implementation
- Terminology import
- Licensed row access
- Runtime DB writes
- External API calls
- UI changes
- Clinical interpretation
- Diagnosis/treatment inference
- Medication/dose/lab value parsing
- DDI behavior
- Real-corpus validation
- Cue expansion

Cue expansion remains NOT recommended.

## Progress Estimate

Whole MedAI done estimate: approximately 95.1%.
