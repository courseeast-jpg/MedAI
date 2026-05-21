# MEDAI-CKA-TERM-INTEGRATION-PRIVATE-ADAPTER-SPEC-01 Report

## Executive Recommendation

Create a future private terminology adapter boundary only as a design SPEC. The selected next block is `CKA-TERM-INTEGRATION-PRIVATE-ADAPTER-READINESS-02`.

Private adapter implementation remains disallowed.

## Why This SPEC Exists After READINESS-03

`CKA-TERM-LICENSE-GATE-READINESS-03` concluded that public-safe evidence is sufficient for a design-only private adapter SPEC. It did not authorize implementation, terminology imports, licensed row access, private store reads, runtime DB access, external APIs, or clinical inference.

## Current Terminology Parked Baseline

- Terminology helper and UI wiring chain: parked
- License gate readiness: ready for private adapter SPEC
- Private adapter implemented: false
- Runtime behavior changed: false
- Licensed terminology rows read: false
- Private store opened or read: false
- License acknowledgement contents read: false
- Cue expansion recommended: false

## Adapter Contract

The future adapter must be an injected, default-off, read-only lookup boundary for private local terminology stores. It must not call external APIs, write DB state, render row content, or perform clinical inference. It may return only aggregate controlled-vocabulary metadata that remains review-bound and public-report safe.

## Input Contract

Allowed input is a controlled candidate query only.

Disallowed inputs include raw OCR text, raw document text, source filenames, private paths, PHI, secrets, free-text clinical interpretation, runtime DB rows, and source/private documents. Raw/private inputs must be rejected or ignored.

## Output Contract

Allowed fields:

- `match_family`
- `terminology_system_family`
- `matches_count`
- `license_class`
- `review_required`
- `auto_accept_allowed`
- `public_report_safe`

Required values:

- `review_required`: true
- `auto_accept_allowed`: false
- `public_report_safe`: true

## Disallowed Output Fields

The adapter must never output row code, display name, synonym, definition, raw concept row, private path, license acknowledgement content, source filename, raw OCR text, raw document text, PHI, or secrets to the public layer.

## License-State Gate

Manual license verification is required before implementation. A future private gate may presence-check `LICENSE_ACK_PRIVATE.json`, but acknowledgement contents must never enter public reports, logs, or UI output.

Missing or unclear license state must fail closed with no output.

## Fail-Closed Behavior

| Condition | Behavior |
| --- | --- |
| Missing adapter | No output |
| Missing license state | No output |
| Unsafe output fields | Reject output |
| Adapter exception | No output and safe diagnostic only |
| Private store missing | No output |
| License state unclear | No output |
| `public_report_safe` false | No output |

## No-Row-Output Rules

Public reports and UI layers may receive aggregate counts and controlled status fields only. They must never receive row content, private paths, filenames, license acknowledgement content, raw text, PHI, or secrets.

## Test Plan

- Synthetic-only unit tests
- Fake private-store adapter contract tests
- Default-off tests
- Missing license state fails-closed tests
- Missing adapter fails-closed tests
- Unsafe row content rejection tests
- Raw/private input rejection tests
- No-row-output tests
- Public report privacy checks
- No runtime DB write tests
- No external API tests
- No clinical inference tests
- No DDI behavior change tests
- Cue expansion remains false tests

## Implementation Prerequisites

- Manual license verification complete
- Private store path configured through local-only config
- License-state gate designed and tested
- Adapter injection boundary defined
- No row content returned to public layer
- Rollback plan defined
- Staged safety gate defined
- Public-report privacy checker extended if needed

## Stop Conditions

Stop before implementation if license verification is incomplete, the private store contract is unclear, adapter output includes row content, public reports include row fields, private paths or secrets appear, a runtime DB write is attempted, an external API call is attempted, clinical inference is introduced, DDI behavior changes, or cue expansion is attempted.

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
- Medication, dose, and lab value parsing
- DDI behavior
- Real-corpus validation
- Cue expansion

Cue expansion remains NOT recommended.

## Safety And Privacy Constraints

This SPEC was reports-only and public-safe. It did not modify runtime code, `app/main.py`, helper code, Streamlit wiring, launchers, startup preflight, config, OCR, extraction, classifier behavior, thresholds, cue packs, clinical logic, DDI behavior, or external API behavior. It did not open private documents, runtime DB rows, licensed terminology rows, private terminology stores, license acknowledgement contents, raw text, private paths, PHI, or secrets.

## Validation Results

| Check | Result |
| --- | --- |
| Public report privacy checks | Passed: 3/3 PRIVATE-ADAPTER-SPEC-01 reports privacy-clean |
| Final CKA MVP validation | Passed: 12/12 validation cases; 693 total tests passed; external API used false |
| B07 term01 validation | Passed: 6/6 cases; external API used false |
| ROUTE-FIX validation | Passed: medai_route_fix01_ready; external API used false |
| UI ops validation | Passed: medai_ui_ops_panel_ready |
| UI boot validation | Passed: medai_ui_boot_fix_startup_resilience_ready |
| Staged safety check | Passed: only PRIVATE-ADAPTER-SPEC-01 report files staged |

## Progress Estimate

Whole MedAI done estimate: approximately 95.1%.
