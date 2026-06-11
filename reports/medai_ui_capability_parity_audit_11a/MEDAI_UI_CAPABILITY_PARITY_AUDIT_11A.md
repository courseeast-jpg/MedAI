# MEDAI-UI-CAPABILITY-PARITY-AUDIT-11A

Current HEAD: `18dd2bf`
Conclusion: `ui_capability_parity_gaps_found_repair_required`

## Summary

- Current UI inventory count: `16`
- Historical capability count: `36`
- Available capability count: `6`
- Hidden/degraded capability count: `6`
- Unexpectedly missing capability count: `2`
- Specialty/domain selector confirmed missing: `True`
- Recommended next block: `MEDAI-UI-CAPABILITY-RESTORE-11B`

## Priority 0 Blockers

- None identified.

## Priority 1 Restorations

- `specialty/domain selector`
- `MKB Explorer`
- `active/quarantined/superseded tiers`
- `specialty routing`

## Feature Parity Matrix

- **document upload/queue** (Primary operator workflow): `available`; action `Keep.`; block `none`
- **start run** (Primary operator workflow): `available/degraded`; action `Clarify degraded labels.`; block `MEDAI-UI-RUN-REVIEW-REPAIR-11B`
- **current run status** (Primary operator workflow): `available`; action `Keep.`; block `none`
- **extracted information preview** (Primary operator workflow): `available`; action `Keep.`; block `none`
- **MKB persistence** (Primary operator workflow): `available`; action `Keep review-bound.`; block `none`
- **operator accept/reject/defer** (Primary operator workflow): `available`; action `Keep no bulk accept.`; block `none`
- **specialty/domain selector** (Primary operator workflow): `unexpectedly missing`; action `Restore explicit domain selector without changing extraction rules.`; block `MEDAI-UI-CAPABILITY-RESTORE-11B`
- **document category selector** (Primary operator workflow): `available`; action `Keep and distinguish from specialty.`; block `MEDAI-UI-CAPABILITY-RESTORE-11B`
- **MKB Explorer** (MKB / knowledge workflow): `unexpectedly missing`; action `Restore safe read-only tab visibility.`; block `MEDAI-UI-CAPABILITY-RESTORE-11B`
- **review-bound records** (MKB / knowledge workflow): `available`; action `Keep.`; block `none`
- **active/quarantined/superseded tiers** (MKB / knowledge workflow): `degraded`; action `Restore MKB Explorer visibility.`; block `MEDAI-UI-CAPABILITY-RESTORE-11B`
- **retrieval proof** (MKB / knowledge workflow): `available`; action `Keep in reports.`; block `none`
- **specialty routing** (MKB / knowledge workflow): `degraded`; action `Restore explicit selector.`; block `MEDAI-UI-CAPABILITY-RESTORE-11B`
- **conflict review** (MKB / knowledge workflow): `disconnected`; action `Restore after MKB Explorer.`; block `MEDAI-UI-CAPABILITY-RESTORE-11C`
- **privacy audit** (Safety / governance): `available`; action `Keep advanced.`; block `none`
- **safe mode** (Safety / governance): `available`; action `Keep.`; block `none`
- **DDI/medication safety** (Safety / governance): `available`; action `Keep.`; block `none`
- **no auto-accept** (Safety / governance): `available`; action `Keep.`; block `none`
- **external API off** (Safety / governance): `available`; action `Keep.`; block `none`
- **audit reports** (Safety / governance): `available`; action `Keep.`; block `none`
- **single-document validation** (Validation / corpus workflow): `available`; action `Keep listed.`; block `none`
- **self-healing validation** (Validation / corpus workflow): `available`; action `Keep.`; block `none`
- **batch audit** (Validation / corpus workflow): `available`; action `Keep advanced.`; block `none`
- **full corpus audit** (Validation / corpus workflow): `available`; action `Keep advanced.`; block `none`
- **OCR review** (Validation / corpus workflow): `available`; action `Keep read-only.`; block `none`
- **unknown triage** (Validation / corpus workflow): `deferred`; action `Defer.`; block `MEDAI-UI-TRIAGE-PARKED`
- **terminology admin** (Terminology workflow): `available`; action `Keep advanced.`; block `none`
- **terminology lookup** (Terminology workflow): `hidden`; action `Keep gated.`; block `none`
- **import readiness** (Terminology workflow): `available`; action `Keep.`; block `none`
- **license-safe reporting** (Terminology workflow): `available`; action `Keep.`; block `none`
- **external agents/comments back** (External connector / agent workflow): `deferred`; action `Defer.`; block `MEDAI-UI-EXTERNAL-AGENTS-DEFERRED`
- **Claude/Gemini status** (External connector / agent workflow): `deferred`; action `Keep disabled.`; block `out-of-scope`
- **cloud API disabled state** (External connector / agent workflow): `available`; action `Keep.`; block `none`
- **future/deferred connector status** (External connector / agent workflow): `available`; action `Keep advanced.`; block `none`

## Safety

- privacy check passed: `True`
- external API used: `False`
- auto-accept enabled: `False`
- No OCR routing, classifier cues, confidence thresholds, DDI, medication, privacy, or review gates were changed.
