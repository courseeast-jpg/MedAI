# MEDAI-V2-RUNTIME-CONTRACTS-01 — Reports-Only / Typing-Only V2 Runtime Interface Contracts

Reports-only / typing-only block. Adds a side-effect-free,
standard-library-only Python contract module plus a focused pytest
module, an audit script, and three public-safe reports. **No** concrete
adapter implementations. **No** runtime wiring. **No** runtime
behavior change. V1 frozen release at `7ef8ffd` remains preserved.
PARK-20..23 / PARK-24..26 / FREEZE / TERM helper/wiring PARK-01 /
license-gate PARK-02 tag pairs all remain intact. Cue expansion
remains explicitly **NOT** recommended.

## A. Scope and non-scope

**In scope**

- Define 10 v2 runtime interface boundaries as typing-only Python
  contracts (Protocol / dataclass / Enum / type aliases).
- Document the boundaries and their invariants in three public-safe
  reports.
- Provide a focused pytest module that proves the contracts module is
  importable, side-effect-free, and standard-library-only.
- Provide an audit script that re-affirms the contract inventory and
  the safety / privacy invariants.

**Out of scope**

- Any concrete adapter implementation.
- Any runtime wiring into `app/main.py`, launchers, startup preflight,
  or config.
- Any extraction / OCR routing / classifier / threshold / cue-pack /
  clinical / DDI behavior change.
- Any private terminology adapter implementation.
- Any external API enablement.
- Any runtime DB access.
- Any modification of pre-existing parking / freeze tags.

## B. Contract inventory (32 contracts across 10 boundaries)

| Boundary | Contracts |
| --- | --- |
| Ingestion | `V2SourceKind`, `V2DocumentSource`, `V2IngestionRequest`, `V2IngestionAdapterProtocol` |
| Text visibility / document quality | `V2VisibilityStatus`, `V2TextVisibilityProfile`, `V2DocumentQualityProtocol` |
| Extraction / OCR orchestration | `V2ExtractionMode`, `V2ExtractionRequest`, `V2ExtractionResult`, `V2ExtractionAdapterProtocol` |
| Document classification | `V2DocumentTypeCandidate`, `V2ClassificationResult`, `V2ClassifierProtocol` |
| Clinical knowledge / terminology | `V2TerminologyQuery`, `V2TerminologyMatchSummary`, `V2TerminologyLookupProtocol` |
| Review / HITL | `V2ReviewDisposition`, `V2ReviewItem`, `V2ReviewQueueProtocol` |
| Operator action | `V2OperatorActionKind`, `V2OperatorActionRequest`, `V2OperatorActionResult`, `V2OperatorActionProtocol` |
| Audit / observability | `V2AuditEventKind`, `V2AuditEvent`, `V2ObservabilitySinkProtocol` |
| Validation report | `V2ValidationStatus`, `V2ValidationReceipt`, `V2ValidationHarnessProtocol` |
| Runtime privacy / safety | `V2RuntimeSafetyProfile`, `V2PrivacyGateProtocol` |

## C. Contract boundary map

- `V2IngestionAdapterProtocol` output describes a `V2DocumentSource`
  that `V2DocumentQualityProtocol` then profiles into a
  `V2TextVisibilityProfile`.
- `V2TextVisibilityProfile` + `V2ExtractionResult` feed
  `V2ClassifierProtocol` to produce a `V2ClassificationResult`.
- `V2ClassificationResult` is review-bound by default; it enters the
  `V2ReviewQueueProtocol` as a `V2ReviewItem`.
- `V2TerminologyMatchSummary` is an aggregate-only optional
  annotation; it must never expose licensed row content.
- `V2OperatorActionProtocol` mutates queue state only; auto-accept is
  forbidden by contract default.
- `V2ObservabilitySinkProtocol` records aggregate `V2AuditEvent`
  instances; no raw text / paths / PHI.
- `V2ValidationHarnessProtocol` carries the v1 five-validation set
  forward unchanged and emits aggregate-only `V2ValidationReceipt`
  instances.
- `V2PrivacyGateProtocol` fails closed on any candidate that contains
  licensed row content, raw text, raw filenames, private paths, PHI,
  or secrets.

## D. Safety / privacy invariants

Default `V2RuntimeSafetyProfile` values:

| Field | Default |
| --- | :-: |
| `local_only` | **true** |
| `review_bound` | **true** |
| `external_api_blocked` | **true** |
| `auto_accept_allowed` | **false** |
| `terminology_lookup_aggregate_only` | **true** |
| `private_adapter_implemented` | **false** |
| `cue_expansion_recommended` | **false** |
| `clinical_decision_expansion` | **false** |

Contract emission rules:

- No raw extracted text in any returned dataclass.
- No raw filename in any returned dataclass.
- No private filesystem path in any returned dataclass.
- No PHI in any returned dataclass.
- No secret in any returned dataclass.
- No licensed terminology row content in any terminology summary.

## E. Terminology / private-adapter boundary

- `V2TerminologyMatchSummary` carries only a controlled-vocabulary
  `match_family`, a controlled-vocabulary `terminology_system_family`,
  an aggregate `matches_count`, and explicit review-bound / refusal /
  privacy invariant flags.
- `licensed_row_content_included` is **false** by contract default.
- `public_report_safe` is **true** by contract default.
- The license-gated private adapter track remains parked at PARK-02
  commit `b9b19ad`. Operator license confirmation status from
  `MEDAI-TERM-LICENSE-OPERATOR-RETURN-08` is `still_required`.
- MeSH status remains `download_helper_created`.

## F. Review-bound default behavior

| Contract field | Default |
| --- | :-: |
| `V2ClassificationResult.review_required` | true |
| `V2ClassificationResult.auto_accept_allowed` | false |
| `V2ClassificationResult.cue_expansion_used` | false |
| `V2ReviewItem.review_required` | true |
| `V2ReviewItem.auto_accept_allowed` | false |
| `V2ReviewItem.disposition` | `PENDING_REVIEW` |
| `V2OperatorActionResult.review_required_after_action` | true |
| `V2OperatorActionResult.state_mutation_performed` | false (for non-state-changing actions) |
| `V2TerminologyMatchSummary.review_required` | true |
| `V2AuditEvent.review_required` | true |
| `V2ValidationReceipt.external_api_used` | false |
| `V2ValidationReceipt.cue_expansion_recommended` | false |

## G. Import / side-effect rules

- Standard library only: `dataclasses`, `datetime`, `enum`, `typing`.
- Forbidden modules: `streamlit`, `requests`, `httpx`, `http`,
  `urllib`, `socket`, `sqlite3`, `sqlcipher3`, any LLM SDK, any
  network client.
- No IO at import time.
- No environment variable reads at import time.
- No filesystem reads at import time.
- No network calls at import time.
- No print at import time.
- No global state mutation at import time.
- No runtime project imports.

A focused test (`test_module_import_is_silent`) captures
`stdout`/`stderr` during import and asserts both are empty. Another
(`test_no_forbidden_imports_in_contracts_module`) static-scans the
module source for forbidden module names.

## H. Validation matrix

| Validation | Result |
| --- | --- |
| Focused V2-RUNTIME-CONTRACTS-01 tests | PASS ✓ |
| Import smoke test | PASS ✓ — module imports with no `stdout`/`stderr` output |
| Audit script (`scripts/run_medai_v2_runtime_contracts_01.py`) | PASS ✓ (`all_clean=true`) |
| Public-report privacy checks (3 reports) | PASS ✓ |
| Final CKA MVP validation | PASS ✓ — `cka_mvp_release_package_ready`, 693 tests, `external_api_used: false` |
| B07 term01 opt-in integration | PASS ✓ — `cases_failed: 0`, `external_api_used: false` |
| ROUTE-FIX 01 | PASS ✓ — `medai_route_fix01_ready`, `passed: true` |
| UI ops panel | PASS ✓ — `medai_ui_ops_panel_ready` |
| UI boot fix | PASS ✓ — `medai_ui_boot_fix_startup_resilience_ready` |
| Staged safety check | PASS — only V2 runtime contracts scoped files staged into the implementation commit |
| Full pytest | Not run. Typing-only / reports-only block; the focused module covers every required invariant. |

## I. Future implementation gates

- Any concrete v2 adapter implementation must be preceded by its own
  approved SPEC block.
- Any concrete v2 adapter implementation must satisfy
  `MEDAI-V2-FOUNDATION-SPEC-02` stop-on-failure rules.
- Any new v2 default-on behavior requires explicit operator approval
  recorded in a separate SPEC.
- Any terminology adapter implementation remains blocked until
  concrete operator license confirmations land (per RETURN-08 /
  PARK-02).
- Any cue-pack work is explicitly **NOT** recommended.
- Any external API enablement requires its own SPEC + readiness
  audit.

## J. Recommended next sequence

1. **`MEDAI-V2-VALIDATION-HARNESS-01`** — reports-only v2 validation
   matrix SPEC plus a focused pytest harness cataloguing the existing
   v1 five-validation set with reports-only stubs for v2 contract
   conformance.
2. **`MEDAI-V2-UI-SHELL-SPEC-01`** or **`MEDAI-V2-DATA-INFRA-SPEC-01`**
   (depending on roadmap signal).
3. **`MEDAI-V2-ROADMAP-02`** — re-evaluate after the validation
   harness and one of UI / data-infra SPECs lands.

V1 frozen release at `7ef8ffd` remains the durable shipped artifact.
Cue expansion remains explicitly **NOT** recommended.
