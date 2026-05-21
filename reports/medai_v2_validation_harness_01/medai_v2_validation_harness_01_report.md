# MEDAI-V2-VALIDATION-HARNESS-01 — Reports-Only V2 Validation Matrix + Contract-Conformance Harness

Reports-only / validation-harness-only block. Catalogs the existing
V1 five-validation health-check set verbatim, defines the V2 8-category
validation matrix, and adds focused contract-conformance tests for
the typing-only V2 runtime contracts. **No** V1 validation script
modified. **No** concrete adapter implemented. **No** runtime wiring.
**No** runtime behavior change. V1 frozen release at `7ef8ffd`
remains preserved. PARK-20..23 / PARK-24..26 / FREEZE / TERM
helper/wiring PARK-01 / license-gate PARK-02 tag pairs all remain
intact. Cue expansion remains explicitly **NOT** recommended.

## A. Scope and non-scope

**In scope**

- Catalog the existing V1 five-validation health-check set verbatim
  (preserved, not new behavior).
- Define the V2 validation matrix across 8 categories (A–H).
- Add contract-conformance focused tests for the typing-only V2
  runtime contracts module.
- Provide a typing-only Python catalog module
  (`clinical_knowledge/v2_contracts/validation_harness.py`)
  containing the V1 catalog and V2 matrix as dataclasses / enums /
  pure helpers.

**Out of scope**

- Any V2 implementation.
- Any runtime wiring.
- Any change to V1 validation scripts.
- Any concrete adapter implementation.
- Any extraction / OCR routing / classifier / threshold / cue /
  clinical / DDI behavior change.
- Any private terminology adapter implementation.
- Any external API enablement.
- Any runtime DB access.
- Any new tag creation or movement.

## B. Prior V2 dependency chain

| Block | Commit |
| --- | --- |
| `MEDAI-V2-ARCHITECTURE-SPEC-01` | `551af98` |
| `MEDAI-V2-FOUNDATION-SPEC-02` | `8b53d82` |
| `MEDAI-V2-RUNTIME-CONTRACTS-01` | `e6e33dd` |

This block satisfies the foundation block's
`validation_harness_only` taxonomy and consumes the typing-only
contracts from the runtime contracts block.

## C. V1 five-validation health-check catalog (preserved)

| Name | Script | Expected conclusion | `external_api_used` |
| --- | --- | --- | :-: |
| `cka_final_mvp_release` | `scripts/run_cka_final_mvp_release_validation.py` | `cka_mvp_release_package_ready` (693 tests) | false |
| `b07_term01_opt_in_integration` | `scripts/run_b07_term01_opt_in_integration_validation.py` | `b07_term01_opt_in_integration_ready` (`cases_failed: 0`) | false |
| `medai_route_fix01` | `scripts/run_medai_route_fix01_validation.py` | `medai_route_fix01_ready` (`passed: true`) | false |
| `medai_ui_ops_panel` | `scripts/run_medai_ui_ops_panel_validation.py` | `medai_ui_ops_panel_ready` | false |
| `medai_ui_boot_fix_startup_resilience` | `scripts/run_medai_ui_boot_fix_validation.py` | `medai_ui_boot_fix_startup_resilience_ready` | false |

This catalog is **preserved verbatim**, not re-implemented. The V1
validation scripts themselves are unchanged by this block.

## D. V2 validation matrix (8 categories)

| Category | Description |
| --- | --- |
| A — Contract import and side-effect safety | runtime contracts import silently; standard-library-only; no Streamlit/network/DB/external/private terminology imports; no side effects at import time |
| B — Contract inventory conformance | 10 boundaries present; 32 contract names resolve; required Protocol classes are protocols; no concrete adapters |
| C — Safety profile conformance | `local_only=True`, `review_bound=True`, `external_api_blocked=True`, `auto_accept_allowed=False`, `terminology_lookup_aggregate_only=True`, `private_adapter_implemented=False`, `cue_expansion_recommended=False`, `clinical_decision_expansion=False` |
| D — Terminology aggregate-only conformance | aggregate `match_family` / `terminology_system_family` / `matches_count` only; `licensed_row_content_included=False`; `public_report_safe=True`; no `code` / `display` / `synonym` / `definition` / `concept` / raw text / OCR text fields |
| E — Review / HITL conformance | `review_required=True` by default; `auto_accept_allowed=False` by default; disposition `PENDING_REVIEW` by default |
| F — Reports privacy conformance | no PHI / raw OCR / raw document text / raw filenames / private filesystem paths / secrets / licensed-row content / license-acknowledgement contents in any public report |
| G — Runtime non-modification conformance | `app/main.py` unchanged; launchers unchanged; startup preflight unchanged; config unchanged; no runtime wiring; no external API used; no runtime DB accessed |
| H — Parking / freeze preservation conformance | V1 frozen release at `7ef8ffd` preserved; PARK-20..23 tag pairs unchanged; PARK-24..26 anchor commits unchanged; term helper/wiring PARK-01 tag pair unchanged; license-gate PARK-02 tag pair unchanged; no new tags created by this block |

Full invariant lists per category live in the JSON under
`section_d_v2_validation_matrix`.

## E. Contract-conformance checks implemented

The focused pytest module
(`tests/test_medai_v2_validation_harness_01.py`) and the audit script
(`scripts/run_medai_v2_validation_harness_01.py`) jointly enforce:

- The validation_harness module imports silently
  (no `stdout` / `stderr`).
- Both `runtime_contracts` and `validation_harness` use only
  standard-library imports.
- The 32 V2 contract names still resolve from
  `clinical_knowledge.v2_contracts.runtime_contracts`.
- The 10 required `Protocol` classes are
  `typing.Protocol`-shaped.
- `V2RuntimeSafetyProfile` defaults carry the foundation invariants.
- `V2TerminologyMatchSummary` is aggregate-only and exposes no
  forbidden row-content fields.
- `V2ReviewItem` defaults are review-bound.
- The V2 validation matrix exposes all 8 categories.

## F. Privacy / report safety checks

All three V2-VALIDATION-HARNESS-01 reports pass
`clinical_knowledge.privacy.check_public_report_payload`. No raw
text, raw OCR text, raw filenames, private filesystem paths, PHI,
secrets, licensed terminology row content, or license-acknowledgement
contents appear in any committed file.

## G. Runtime non-modification checks

- `app/main.py` source does not mention
  `MEDAI-V2-VALIDATION-HARNESS-01` or import
  `v2_contracts.validation_harness`.
- The four launchers
  (`Start_MedAI_UI.bat`, `Start_MedAI_UI_Silent.vbs`,
  `Start_MedAI_Test_UI.bat`, `Start_MedAI_UI_Encrypted.bat`)
  unchanged.
- `app/startup_preflight.py` unchanged.
- `app/config.py` unchanged.
- No runtime wiring added.
- No external API used.
- No runtime DB accessed.

## H. Parking / freeze preservation

| Track | Status | Anchor |
| --- | --- | --- |
| Local operator release | **frozen** | `7ef8ffd` |
| Residual Unknown reduction | parked | `3e46461` |
| Text-layer eval spec | parked | `9f9e22d` |
| PDF text/layout default-off | parked | `f4d3cc6` |
| PDF text/layout Streamlit wiring | parked | `748c32a` |
| DIAG-20 operator UAT | parked | `1b14ffe` |
| DIAG-21 fixture audit | parked | `6b31678` |
| Operator readiness + runtime hardening | parked | `91b9eba` |
| Terminology helper/wiring mini-track | parked | `e398a75` |
| License-gated private adapter track | parked | `b9b19ad` |
| Private terminology config boundary | complete & verified | `60f1114` |
| MeSH local helper / download | blocked operator-side | `cedbbd3` |
| Manual license verification gate | blocked operator-side | `376ca4e` |

## I. Stop-on-failure rules inherited from V2-FOUNDATION-SPEC-02

- Stop if any prior parking / freeze tag is moved, deleted,
  repointed, or duplicated under a different SHA.
- Stop if runtime files are modified in a validation-harness-only
  block.
- Stop if any test would require licensed terminology rows, private
  DB rows, or private filesystem paths.
- Stop if any test would require Streamlit at module load in an
  environment where Streamlit is unavailable.
- Stop if any new validation overwrites or modifies an existing v1
  validation script.
- Stop if any external API is detected at runtime.
- Stop if private / source documents appear in `git status` or
  staged files.
- Stop if a future block attempts implementation without a preceding
  approved SPEC.

## J. Recommended next block sequence

Choose **one** of:

- **`V2-UI-SHELL-SPEC-01`** — reports-only UI shell SPEC; documents
  the existing Run & Review + Advanced technical details surface and
  defines env-gating contracts for any future UI expansion.
- **`V2-DATA-INFRA-SPEC-01`** — reports-only DB / persistence SPEC;
  defines a runtime-DB-row-blind contract with explicit rollback
  expectations.
- **`V2-EXTRACTION-SPEC-01`** — reports-only extraction / OCR layer
  SPEC; only if extraction architecture planning is explicitly
  selected.

Forbidden actions in any of these blocks:

- Start V2 implementation directly.
- Add runtime wiring from this harness.
- Convert harness stubs into real adapters.
- Reopen terminology / private adapter implementation.
- Reopen cue expansion (explicitly **NOT** recommended).

V1 frozen release at `7ef8ffd` remains the durable shipped artifact.
