# MEDAI-V2-DATA-INFRA-SPEC-01 — Reports-Only V2 Data / Persistence Architecture Spec

Reports-only data-infra SPEC. Defines the future V2 data /
persistence boundaries, the runtime-DB-row-blind doctrine, the
conceptual persistence store map, the ledger / audit separation,
rollback expectations, migration gates, and the public-report data
doctrine. **No** DB schema change. **No** migration. **No** runtime
DB row read. **No** persistence code change. **No** UI / Streamlit
change. **No** runtime behavior change. V1 frozen release at
`7ef8ffd` remains preserved. PARK-20..23 / PARK-24..26 / FREEZE /
TERM helper-wiring PARK-01 / license-gate PARK-02 tag pairs all
remain intact. Cue expansion remains explicitly **NOT** recommended.

## A. Scope and non-scope

**In scope**

- Define future V2 data / persistence boundaries at planning level
  only.
- Establish the runtime-DB-row-blind doctrine for every subsequent
  block.
- Map the conceptual persistence stores (no schema, no migrations).
- Define ledger / audit separation rules.
- Define rollback expectations.
- Define migration gates.
- Define public-report data doctrine.
- Preserve the terminology / private-adapter data boundary.
- Preserve the V1 five-validation health-check catalog reference.

**Out of scope**

- Any DB schema change.
- Any migration (forward or rollback).
- Any persistence code change.
- Any runtime DB row read.
- Any change to `app/main.py`, launchers, startup preflight, or
  config.
- Any UI / Streamlit code change.
- Any extraction / OCR routing / classifier / threshold / cue /
  clinical / DDI behavior change.
- Any private terminology adapter implementation.
- Any external API enablement (cloud, remote DB, hosted storage).
- Any tag creation or movement.

## B. Prior V2 dependency chain

| Block | Commit |
| --- | --- |
| `MEDAI-V2-ARCHITECTURE-SPEC-01` | `551af98` |
| `MEDAI-V2-FOUNDATION-SPEC-02` | `8b53d82` |
| `MEDAI-V2-RUNTIME-CONTRACTS-01` | `e6e33dd` |
| `MEDAI-V2-VALIDATION-HARNESS-01` | `73af6f6` |
| `MEDAI-V2-UI-SHELL-SPEC-01` | `745a980` |

## C. Runtime-DB-row-blind doctrine

- Runtime DB is private operational state.
- Reports must be runtime-DB-row-blind.
- No V2 report may contain: DB rows, raw records, document
  filenames, raw OCR text, raw document text, PHI, private filesystem
  paths, licensed terminology rows, license-acknowledgement contents,
  private config contents, secrets / API keys / credentials.
- Future DB access requires a separate, privacy-approved
  implementation SPEC.
- Metadata-only diagnostics remain acceptable (counts, controlled-
  vocabulary statuses, anonymous IDs).

## D. Conceptual persistence store map (8 future stores)

| ID | Store | Purpose | Implemented in this SPEC |
| :-: | --- | --- | :-: |
| PS1 | `document_registry_metadata_store` | track local session documents anonymously | no |
| PS2 | `extraction_result_store` | hold extraction outputs locally; no row emission | no |
| PS3 | `review_queue_store` | hold review-bound items until operator disposition | no |
| PS4 | `operator_action_ledger` | append-only record of operator dispositions | no |
| PS5 | `validation_receipt_store` | hold V1 health-check receipts + future V2 receipts | no |
| PS6 | `audit_observability_event_store` | aggregate-only audit events; no PHI | no |
| PS7 | `terminology_aggregate_cache_if_later_approved` | future aggregate-only cache; gated by license clearance | no (blocked) |
| PS8 | `quarantine_blocked_item_store_if_later_approved` | future quarantine for blocked items; no row emission | no |

Each store is conceptual only. No schema, no migration, no code in
this block. Each store is row-blind in public reports.

## E. Ledger / audit separation

- Operator action ledger is **separate from** runtime event audit.
- Validation receipts are **separate from** audit events.
- Privacy audit receipts are **separate from** validation receipts.
- Release / parking snapshots are **separate from** all ledgers.
- All ledgers are **append-only** by default.
- Any non-append behavior requires a separate rollback SPEC.

## F. Rollback doctrine

- Every schema-changing block requires a pre-migration snapshot.
- Migration must be reversible **or** explicitly one-way with
  documented justification.
- Rollback verification must be tested (synthetic data only inside
  any SPEC block).
- Public reports contain only aggregate rollback status.
- No DB rows or private paths in rollback reports.
- Rollback failure handling requires a documented recovery path
  before the migration is approved.

## G. Migration gate doctrine (9 gates)

1. data-infra SPEC complete
2. runtime contract alignment complete
3. validation harness updated for the new store / field
4. privacy scanner updated if a new field class is introduced
5. rollback plan created
6. dry-run migration tested on synthetic data only
7. staged safety check clean
8. operator approval recorded in a public-safe report
9. real runtime DB access still blocked until the implementation
   block

All nine must clear before any data migration block lands.

## H. Public-report data doctrine

| Allowed | Prohibited |
| --- | --- |
| counts | PHI |
| booleans | raw OCR text |
| controlled-vocabulary statuses | raw document text |
| anonymous IDs / hashes where already approved | raw filenames |
| validation names | private filesystem paths |
| short SHAs (~7 chars max) | runtime DB rows |
| aggregate result summaries | licensed terminology row values |
| | license-acknowledgement contents |
| | private config contents |
| | secrets / API keys / credentials |

Long 40-char SHAs trigger the privacy scanner (known constraint
from prior chain).

## I. Terminology / private-adapter data boundary

- Licensed terminology row access remains **blocked**.
- Private adapter implementation remains **blocked**.
- Terminology output remains aggregate-only.
- Public reports show only license-gate status, not licensed
  content.
- MeSH integration remains **blocked** until operator-side
  license / download conditions are satisfied.
- PARK-02 anchor commit: `b9b19ad`.
- PARK-01 anchor commit: `e398a75`.
- Operator license confirmation status (from RETURN-08):
  `still_required`.
- MeSH status: `download_helper_created`.

## J. Validation / health persistence boundary

The V1 five-validation health-check set is preserved verbatim:

- `cka_final_mvp_release`
- `b07_term01_opt_in_integration`
- `medai_route_fix01`
- `medai_ui_ops_panel`
- `medai_ui_boot_fix_startup_resilience`

Any future validation receipt store must:

- Be aggregate-only.
- Not embed raw test output.
- Consume `V2ValidationHarnessProtocol` from `RUNTIME-CONTRACTS-01`.

## K. Safety / privacy invariants

| Invariant | Value |
| --- | :-: |
| `block_mode` | `reports_only_data_infra_spec` |
| `runtime_db_row_blind` | **true** |
| `runtime_db_accessed` | **false** |
| `schema_changed` | **false** |
| `migration_created` | **false** |
| `migration_executed` | **false** |
| `persistence_code_changed` | **false** |
| `runtime_behavior_changed` | **false** |
| `app_main_changed` | **false** |
| `streamlit_code_changed` | **false** |
| `ui_changed` | **false** |
| `launcher_changed` / `startup_config_changed` | **false** |
| `extraction_changed` / `ocr_changed` / `classifier_changed` / `threshold_scoring_changed` / `cue_pack_changed` | **false** |
| `clinical_behavior_changed` / `ddi_behavior_changed` / `terminology_behavior_changed` | **false** |
| `private_adapter_implemented` / `concrete_adapters_implemented` / `runtime_wiring_added` | **false** |
| `external_api_used` | **false** |
| `private_data_accessed` / `licensed_rows_read` / `licensed_rows_exposed` / `private_license_ack_read` / `private_config_read` / `source_documents_opened` | **false** |
| `raw_text_printed` / `raw_filenames_printed` / `private_paths_printed` / `secrets_printed` | **false** |
| `tags_touched` | **false** |
| `cue_expansion_recommended` | **false** |
| `v1_release_preserved` | **true** |

## L. Future implementation gates

A. Any schema change requires a separate implementation block.
B. Any migration requires a pre-migration snapshot.
C. Any migration requires a rollback plan.
D. Any migration requires a synthetic-data dry run before any real
   data run.
E. Any real runtime DB access requires explicit operator approval
   recorded in a public-safe report.
F. Any private DB / report bridge must pass the privacy scanner
   review (`clinical_knowledge.privacy.check_public_report_payload`).
G. Any terminology cache must remain aggregate-only unless the
   license gate changes via a separately approved SPEC.
H. Any validation receipt store must not expose PHI, private paths,
   raw OCR text, raw document text, raw filenames, or licensed row
   content.
I. Any UI display of persistence state must be covered by a
   separate UI implementation block (mapped to V2-UI-SHELL-SPEC-01
   + its future implementation SPEC).
J. Any external storage / cloud persistence requires a privacy /
   safety SPEC and must remain blocked by default.

## M. Validation matrix

V2 validation matrix categories referenced (all 8 from
`V2-VALIDATION-HARNESS-01`):

| Category | Conformance |
| --- | --- |
| A — Contract import + side-effect safety | no module side effects added |
| B — Contract inventory conformance | no contract change |
| C — Safety profile conformance | default safety profile unchanged |
| D — Terminology aggregate-only conformance | terminology aggregate-only doctrine preserved |
| E — Review / HITL conformance | review-bound default preserved |
| F — Reports privacy conformance | 3 reports pass privacy check |
| G — Runtime non-modification conformance | no runtime / `app/main.py` / launcher / preflight / config / Streamlit change |
| H — Parking / freeze preservation conformance | all 10 pre-existing tag groups unchanged |

## N. Recommended next block

- **Primary:** `V2-EXTRACTION-SPEC-01` — reports-only extraction /
  OCR layer SPEC; only if extraction architecture planning is the
  priority.
- **Checkpoint alternative:** `V2-ROADMAP-02` — reports-only roadmap
  audit after the foundation / contracts / validation-harness /
  UI-shell / data-infra sequence; re-ranks the remaining v2
  workstreams.
- **Packaging alternative:** `V2-PACKAGING-SPEC-01` — reports-only
  packaging / deployment SPEC if operator deployment is the
  priority.

Forbidden in any of these blocks:

- Start V2 data implementation directly.
- Modify DB / schema / migrations.
- Access runtime DB rows.
- Reopen terminology / private adapter implementation.
- Reopen cue expansion (explicitly **NOT** recommended).
- Touch any existing parking / freeze tag.

V1 frozen release at `7ef8ffd` continues to be the durable shipped
artifact.
