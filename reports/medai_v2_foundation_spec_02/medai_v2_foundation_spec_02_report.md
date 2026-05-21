# MEDAI-V2-FOUNDATION-SPEC-02 — Reports-Only V2 Foundation Control Spec

Reports-only foundation SPEC. Converts the v2 architecture plan from
`MEDAI-V2-ARCHITECTURE-SPEC-01` into a durable foundation doctrine —
invariants, termination rules, block taxonomy, safety boundaries,
privacy gates, and clean sequencing rules. No runtime change. No tags
created. PARK-20..23 / PARK-24..26 / FREEZE / TERM helper/wiring
PARK-01 / license-gate PARK-02 tag pairs all remain intact. Cue
expansion remains explicitly **NOT** recommended.

## A. V2 foundation principles

1. Frozen v1 local operator release at commit `7ef8ffd` (FREEZE tag
   pair) remains the safe shipped baseline.
2. v2 is contract-first and spec-first.
3. No direct v2 implementation may start without a preceding SPEC
   block.
4. All changes must preserve local-only, review-bound, default-off
   posture unless a later approved SPEC explicitly changes that.
5. Clinical behavior remains safety-gated.
6. Cue expansion remains explicitly **NOT** recommended.
7. Public reports must remain aggregate-only and privacy-clean.
8. Every high-risk track ends with a parking snapshot (one commit +
   two annotated tags).
9. Existing parking tags (PARK-20..26, FREEZE, term helper/wiring
   PARK-01, license-gate PARK-02) remain unchanged across every
   subsequent block.
10. v1.1 trust hierarchy (privacy boundary, MKB local-only store,
    plugin registry, self-enriching knowledge loop) remains immutable
    in v2 design until a separate approved SPEC reopens any of these.

## B. Required invariants for all future V2 blocks

| Invariant | Value |
| --- | :-: |
| `local_only_default` | **true** |
| `external_api_default_blocked` | **true** |
| `review_bound_default` | **true** |
| `no_auto_accept_without_spec` | **true** |
| `no_private_data_access_without_spec` | **true** |
| `no_licensed_row_access_without_operator_license_gate` | **true** |
| `no_runtime_db_migration_without_rollback_spec` | **true** |
| `no_clinical_inference_expansion_without_safety_spec` | **true** |
| `cue_expansion_recommended` | **false** |
| `v1_release_preserved` | **true** |
| `aggregate_only_public_reports_required` | **true** |
| `public_report_privacy_check_required_before_commit` | **true** |
| `staged_safety_check_required_before_commit` | **true** |
| `focused_pytest_module_required_per_implementation_block` | **true** |

## C. Block taxonomy

### Allowed block classes

| Class | Definition | Requires preceding SPEC |
| --- | --- | :-: |
| `reports_only_spec` | Pure planning / SPEC; public-safe reports only. | No |
| `contract_stub_only` | Typing-only Python contracts (`typing.Protocol` / `dataclass`); no runtime side effects on import. | **Yes** |
| `validation_harness_only` | Focused pytest harness; v1 five-validation set carried unchanged. | **Yes** |
| `default_off_helper` | Default-off, fail-closed env-gated helper; aggregate-only metadata; review-bound. | **Yes** |
| `default_off_ui_surface` | Default-off, read-only UI inside Advanced technical details; only `st.markdown` / `st.caption`. | **Yes** |
| `data_migration_spec_only` | DB / persistence SPEC with explicit rollback plan; no actual migration. | No |
| `operator_uat_receipt` | Reports-only UAT against frozen v1 or default-off helper under synthetic load. | **Yes** |
| `parking_snapshot` | Reports-only + tags-only park (1 commit + 2 annotated tags). | No |
| `release_freeze_snapshot` | Reports-only + tags-only release freeze (2 annotated tags). | **Yes** |

### Disallowed block classes (unless separately approved)

- `direct_runtime_rewrite`
- `direct_extraction_behavior_change`
- `direct_ocr_routing_change`
- `direct_threshold_change`
- `direct_classifier_behavior_change`
- `direct_cue_pack_expansion`
- `direct_private_adapter_implementation`
- `direct_licensed_terminology_row_access`
- `direct_external_api_runtime_integration`
- `direct_clinical_decision_logic_expansion`
- `direct_runtime_db_migration`

## D. Stop-on-failure rules

A future block must **stop immediately** if any of the following hold:

- Branch is dirty before starting, unless dirty files are known
  validation receipts isolated to a receipt-refresh commit.
- Private / source documents appear in `git status`.
- `LICENSE_ACK_PRIVATE.json`, terminology rows, runtime DB files, raw
  OCR text, raw document text, raw filenames, private filesystem
  paths, secrets, or PHI appear in staged files or in any public
  report.
- Runtime files are modified in a reports-only block.
- `app/main.py` is modified in a reports-only block.
- Extraction / OCR routing / classifier / threshold / cue / clinical
  / DDI behavior changes during any block other than a directly-
  scoped behavior-change SPEC and its own approved implementation.
- External API use is detected at runtime
  (`external_api_used_for_runtime` must remain false unless an
  explicitly approved SPEC enables it).
- Focused tests show safety / privacy invariant regression.
- A future block attempts implementation without a preceding approved
  SPEC.
- A parking / freeze tag is moved, deleted, repointed, or duplicated
  under a different SHA.
- `git push --tags` or `--force` / `--force-with-lease` is used.

## E. Future sequence rules

Recommended next sequence after this block:

1. `MEDAI-V2-RUNTIME-CONTRACTS-01`
2. `MEDAI-V2-VALIDATION-HARNESS-01`
3. `MEDAI-V2-UI-SHELL-SPEC-01` **or** `MEDAI-V2-DATA-INFRA-SPEC-01`
   (depending on roadmap signal)

**Must not combine:**
- Runtime contracts with implementation.
- Validation harness with real runtime migration.
- Reports-only blocks with runtime-behavior-changing work.
- License-gated terminology work with anything else.
- Cue expansion with anything (cue expansion remains explicitly
  **NOT** recommended).
- Private adapter implementation with any other work.

**Must not reopen without an explicit signal:**
- Terminology / private adapter work until operator license return is
  complete.
- `MORE-UNKNOWN-DIAGNOSTICS` unless a fresh operator-UAT failure
  signal exists.
- `CUE-EXPANSION` (explicitly **NOT** recommended).

## F. Relationship to prior architecture / SPEC documents

v1.1 immutable constraints preserved:

- Local-only MKB.
- PII stripping before any external call.
- Plugin registry.
- Self-enriching knowledge loop.
- Trust hierarchy immutability.
- Progressive activation via feature flags (not uncontrolled
  activation).

v2 strategic intent documented only — no implementation:

- Multi-tier / multi-model architecture is a v2 design goal; this
  block does not implement it.
- v2 architecture goals from `MEDAI-V2-ARCHITECTURE-SPEC-01` remain
  valid.

v2 workstream split remains valid:

- `MEDAI-V2-FOUNDATION-SPEC-02` (this block)
- `MEDAI-V2-RUNTIME-CONTRACTS-01`
- `MEDAI-V2-VALIDATION-HARNESS-01`
- `MEDAI-V2-UI-SHELL-SPEC-01`
- `MEDAI-V2-DATA-INFRA-SPEC-01`
- `MEDAI-V2-EXTRACTION-SPEC-01`
- `MEDAI-V2-TERMINOLOGY-WAIT-GATE`
- `MEDAI-V2-PACKAGING-SPEC-01`
- `MEDAI-V2-ROADMAP-02`

## G. Parked / frozen track preservation table

| Track | Status | Anchor commit | Tag pair |
| --- | --- | --- | --- |
| Local operator release | **frozen** | `7ef8ffd` | `medai-local-operator-release-frozen-2026-05-20`, `medai-final-local-operator-release-2026-05-20` |
| Residual Unknown reduction | parked | `3e46461` | `medai-unknown-diag-language-metadata-ready-2026-05-19`, `medai-final-parked-post-unknown-diag-language-metadata-2026-05-19` |
| Text-layer eval spec | parked | `9f9e22d` | `medai-text-layer-eval-spec-ready-2026-05-19`, `medai-final-parked-post-diag-16-2026-05-19` |
| PDF text/layout default-off | parked | `f4d3cc6` | `medai-pdf-text-layout-quality-default-off-ready-2026-05-19`, `medai-final-parked-post-diag-18-2026-05-19` |
| PDF text/layout Streamlit wiring | parked | `748c32a` | `medai-pdf-text-layout-quality-streamlit-wiring-ready-2026-05-19`, `medai-final-parked-post-diag-19-2026-05-19` |
| DIAG-20 operator UAT | parked | `1b14ffe` | `medai-pdf-text-layout-quality-env-on-uat-ready-2026-05-19`, `medai-final-parked-post-diag-20-2026-05-19` |
| DIAG-21 fixture audit | parked | `6b31678` | `medai-streamlit-fixture-audit-ready-2026-05-19`, `medai-final-parked-post-diag-21-2026-05-19` |
| Operator readiness + runtime hardening | parked | `91b9eba` | `medai-operator-runtime-readiness-ready-2026-05-20`, `medai-final-parked-post-runtime-hardening-2026-05-20` |
| Terminology helper/wiring mini-track | parked | `e398a75` | `medai-cka-term-helper-wiring-ready-2026-05-20`, `medai-final-parked-post-term-wiring-2026-05-20` |
| License-gated private adapter track | parked | `b9b19ad` | `medai-cka-term-license-gated-adapter-parked-2026-05-21`, `medai-final-parked-post-term-license-gate-2026-05-21` |
| Private terminology config boundary | complete & verified | `60f1114` | — |
| MeSH local helper / download | blocked operator-side | `cedbbd3` | — |
| Manual license verification gate | blocked operator-side | `376ca4e` | — |

## Safety / privacy confirmation

- `runtime_behavior_changed` / `app_main_changed` / `ui_changed` /
  `launcher_changed` / `startup_config_changed`: **false**.
- `extraction_changed` / `ocr_changed` / `classifier_changed` /
  `threshold_scoring_changed` / `cue_pack_changed`: **false**.
- `clinical_behavior_changed` / `ddi_behavior_changed` /
  `terminology_behavior_changed`: **false**.
- `private_adapter_implemented` / `external_api_used` /
  `private_data_accessed` / `licensed_rows_read` /
  `private_license_ack_read` / `private_config_read` /
  `runtime_db_accessed` / `source_documents_opened`: **false**.
- `raw_text_printed` / `raw_filenames_printed` /
  `private_paths_printed` / `secrets_printed`: **false**.
- `tags_touched` / `tags_created` / `tags_modified` /
  `prior_park_tags_touched`: **false**.
- `cue_expansion_recommended`: **false**.
- `v1_release_preserved`: **true**.

## Recommended next block

**`MEDAI-V2-RUNTIME-CONTRACTS-01`** — reports-only contracts (Python
`typing.Protocol` / `dataclass` only) for v2 runtime interfaces
(ingestion, extraction adapter, classifier adapter, terminology
lookup adapter, persistence adapter, observability hook). No actual
implementations.

## Progress estimate

| Track | Before this block | After this block |
| --- | --- | --- |
| Whole MedAI project | ~96.1% done / ~3.9% remaining | ~96.2% done / ~3.8% remaining |

Foundation doctrine is now durable. No runtime change. V1 frozen
release remains preserved.
