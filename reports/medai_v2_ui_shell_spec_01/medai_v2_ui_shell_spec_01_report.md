# MEDAI-V2-UI-SHELL-SPEC-01 — Reports-Only V2 Operator UI Shell Planning Spec

Reports-only / ui-shell-spec block. Defines the future V2 operator-
facing UI shell at planning level only: 8 shells, safety / privacy
invariants, the read-only Advanced technical details doctrine, the
terminology / private-adapter UI wait gate, the parking / freeze
visibility doctrine, and 10 future UI implementation gates. **No**
Streamlit code change. **No** `app/main.py` change. **No** UI buttons /
forms / callbacks / session-state / operator actions. **No** runtime
behavior change. V1 frozen release at `7ef8ffd` remains preserved.
PARK-20..23 / PARK-24..26 / FREEZE / TERM helper-wiring PARK-01 /
license-gate PARK-02 tag pairs all remain intact. Cue expansion
remains explicitly **NOT** recommended.

## A. Scope and non-scope

**In scope**

- Define the V2 operator UI shell screen-by-screen at planning level
  only.
- Document safety / privacy UI invariants the future V2 UI must
  satisfy.
- Document the read-only Advanced technical details doctrine carried
  forward from DIAG-19.
- Document the terminology / private-adapter UI wait gate.
- Document the parking / freeze visibility doctrine.
- Document explicit future UI implementation gates that must be
  satisfied before any UI code lands.

**Out of scope**

- Any Streamlit code change.
- Any change to `app/main.py`.
- Any change to launcher / startup preflight / config / helper
  modules.
- Any UI button / form / callback / session-state /
  `on_click` / `on_change` / `on_submit` / state mutation / data-layer
  write addition.
- Any operator action wiring (approve / reject / re-extract /
  escalate).
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
| `MEDAI-V2-VALIDATION-HARNESS-01` | `73af6f6` |

## C. Existing V1 operator UI posture

- Streamlit-based Run & Review surface in `app/main.py` with Advanced
  technical details expander.
- Default-off env-gated read-only metadata blocks established by
  DIAG-17 / DIAG-18 / DIAG-19 and the terminology helper / wiring
  mini-track (PARK-01 at `e398a75`).
- v1 frozen at `7ef8ffd`.
- Shipped local-only environment defaults: `MEDAI_LOCAL_ONLY=1`,
  `MEDAI_ALLOW_EXTERNAL_API=0`, `MEDAI_REQUIRE_PII_SCRUB=1`,
  `MEDAI_PRIVACY_AUDIT=1`.
- Default-off env vars for v1 optional metadata:
  `MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_IMPL_ENABLED`,
  `MEDAI_DOC_TYPE_PDF_TEXT_LAYOUT_QUALITY_UI_ENABLED`,
  `MEDAI_TERMINOLOGY_LOOKUP_ENABLED`.
- V1 review-bound, read-only Advanced-technical-details, no-auto-
  accept invariants preserved.

## D. Proposed V2 UI shell map (8 shells)

| Shell | Purpose | Key invariants |
| --- | --- | --- |
| **S1 Operator Home / Current Run** | Show current local session state | Distinguish `no_current_run`, `previous_run`, `queued_files`, `active_processing`, `blocked`; no raw filenames in public reports; local-only; review-bound |
| **S2 Run & Review** | Operator review of processed documents | `review_required` default true; `auto_accept_allowed` false; visually distinct workflow states (`accepted` / `review` / `blocked` / `error`); raw technical details remain behind Advanced expander |
| **S3 Advanced Technical Details** | Read-only diagnostic visibility | Collapsed by default; no action buttons in diagnostic regions; carries forward v1 default-off metadata surfaces (DIAG-17/18/19, terminology helper/wiring); must not become a hidden automation control plane |
| **S4 Safety / Privacy Status** | Visible local-only / external-API-blocked / privacy-audit / review-bound status | Makes unsafe / degraded states visible; must not imply medical-device status; preserves decision-support-only posture |
| **S5 Validation / Health** | Show V1 five-validation health-check + future V2 validation matrix status | Read-only in this spec; no UI-triggered validation runs; no runtime callbacks |
| **S6 Terminology / Clinical Knowledge** | Show terminology / private-adapter wait gate status only | No licensed rows; no private config contents; no license-acknowledgement contents; shows private adapter implementation blocked |
| **S7 Parking / Release** | Show frozen release + parked track anchors | V1 frozen release at `7ef8ffd` preserved; no tag creation / movement / deletion; does not imply parked tracks are reopened |
| **S8 Operator Guidance** | Plain-language guidance on what the operator can safely do | Separates `safe_to_use_now`, `review_required`, `blocked`, `not_implemented`; does not recommend cue expansion |

No shell adds action buttons, callbacks, or session-state in this
SPEC. Future implementation blocks must consume the relevant V2
Protocol from `RUNTIME-CONTRACTS-01` (e.g.
`V2OperatorActionProtocol`, `V2ValidationHarnessProtocol`,
`V2ReviewQueueProtocol`).

## E. Screen / surface inventory

- 8 shells defined (S1–S8).
- 0 action buttons added in this spec.
- 0 callbacks added in this spec.
- 0 session-state keys added in this spec.

## F. Safety / privacy UI invariants

| Invariant | Value |
| --- | :-: |
| `ui_spec_only` | **true** |
| `streamlit_code_changed` | **false** |
| `app_main_changed` | **false** |
| `runtime_behavior_changed` | **false** |
| `no_ui_actions_added` | **true** |
| `no_callbacks_added` | **true** |
| `no_session_state_logic_added` | **true** |
| `read_only_shell_spec` | **true** |
| `review_bound_default` | **true** |
| `local_only_default` | **true** |
| `external_api_blocked_default` | **true** |
| `auto_accept_allowed_default` | **false** |
| `private_adapter_implemented` | **false** |
| `licensed_rows_exposed` | **false** |
| `raw_filenames_exposed` | **false** |
| `private_paths_exposed` | **false** |
| `cue_expansion_recommended` | **false** |
| `v1_release_preserved` | **true** |

## G. Read-only Advanced technical details doctrine

- Default collapsed.
- Controlled-vocabulary emission only.
- Safe Streamlit calls when implemented later: only `st.markdown` and
  `st.caption`.
- Forbidden widgets in diagnostic regions: `st.button`, `st.form`,
  `st.checkbox`, `st.radio`, `st.selectbox`, `st.text_input`,
  `st.text_area`, `st.number_input`, `st.session_state`,
  `st.rerun`, `st.experimental_rerun`, `st.form_submit_button`.
- Forbidden kwargs in diagnostic regions: `on_click=`, `on_change=`,
  `on_submit=`.
- Forbidden render-field substrings: `on_click`, `on_change`,
  `on_submit`, `button`, `callback`, `action`, `write`, `mutate`,
  `document_type_mutation`, `data_layer_write`, `state_mutation`.
- Must not become a hidden automation control plane.
- The V1 DIAG-19 wiring doctrine is carried forward unchanged.

## H. Terminology / private-adapter UI wait gate

- `private_adapter_implementation_allowed`: **false**
- `real_private_store_access_allowed`: **false**
- `manual_license_verification_complete`: **false**
- `license_gated_resources_verified_count`: **0**
- `internal_boundaries_verified_count`: **2**
- PARK-02 anchor commit: `b9b19ad`.
- PARK-01 anchor commit: `e398a75`.
- UI must not expose licensed rows.
- UI must not expose private config contents.
- UI must not read license-acknowledgement contents.
- UI must show implementation-blocked message only.
- UI emission must remain aggregate-only.

## I. Parking / freeze visibility doctrine

| Track | Status | Anchor |
| --- | --- | --- |
| Local operator release | **frozen** | `7ef8ffd` (FREEZE tag pair) |
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

The Parking / Release shell (S7) must make all 13 entries visible
without implying reopening, and must never display the actual
operator-side filesystem path for any private artifact.

## J. Future UI implementation gates

A. UI implementation requires a separate implementation block.
B. Any UI implementation must be default-off if it changes
   operator-visible behavior.
C. Any action button requires a separate operator-action contract
   review (mapped to `V2OperatorActionProtocol` from
   `RUNTIME-CONTRACTS-01`).
D. Any callback / session-state logic requires focused tests proving
   review-bound default, no auto-accept, no DDI / clinical-decision
   drift.
E. Any private / terminology display must remain aggregate-only and
   license-gated.
F. Any validation-runner UI must be separate from this shell spec
   and must consume `V2ValidationHarnessProtocol` from
   `RUNTIME-CONTRACTS-01`.
G. Any external API toggle must remain blocked by default and require
   a privacy / safety SPEC.
H. Any clinical display must preserve decision-support-only language.
I. Any raw document display requires a separate privacy-approved
   design.
J. Cue expansion remains explicitly **NOT** recommended.

## K. Validation matrix

- V1 five-validation health-check set preserved (no v1 script
  modified).
- V2 validation-harness 8-category matrix applies; this block
  conforms to all 8 categories (see JSON
  `section_k_validation_matrix_reference`).
- Focused tests + audit script verify report invariants, prior-V2-
  dependency presence, no UI / runtime modification, and privacy
  checks pass on all three reports.

## L. Recommended next block

- **Primary:** `V2-DATA-INFRA-SPEC-01` — reports-only DB / persistence
  SPEC; defines a runtime-DB-row-blind contract with explicit
  rollback expectations.
- **Alternative (if extraction planning selected):**
  `V2-EXTRACTION-SPEC-01` — reports-only extraction / OCR layer SPEC.
- **Alternative (if checkpoint preferred):** `V2-ROADMAP-02` —
  reports-only roadmap audit after the
  foundation / contracts / validation-harness / UI-shell sequence;
  re-ranks the remaining v2 workstreams.

Forbidden in any of these blocks:

- Start V2 UI implementation directly.
- Modify `app/main.py`.
- Add Streamlit UI code.
- Add buttons, callbacks, or operator actions.
- Reopen terminology / private adapter implementation.
- Reopen cue expansion.
- Touch any existing parking / freeze tag.

V1 frozen release at `7ef8ffd` continues to be the durable shipped
artifact.
