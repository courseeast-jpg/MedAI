# MEDAI-TERM-LICENSE-MANUAL-RETURN-05 — Provisional Public-Safe License Verification Return

Reports-only provisional return. Packages the current public-safe
evidence for each license-gated terminology resource into a machine-
readable receipt **without** claiming legal / license verification is
complete. No runtime change. No tags created. PARK-20..23 tag pairs
and the FREEZE tag pair remain intact. Cue expansion remains
explicitly **NOT** recommended.

## 1. Executive summary

`manual_license_verification_complete` = **false**.
`manual_license_verification_status` = **`provisional_evidence_present_still_required`**.

Across all four prior public-safe terminology blocks
(`CKA-TERM-LICENSE-MANUAL-RETURN-PACK-04`, `TERM-PRIVATE-CONFIG-01`,
`TERM-PRIVATE-CONFIG-VERIFY-02`, `TERM-MESH-LOCAL-DOWNLOAD-PACK-01`)
there is **no explicit operator return** that confirms license terms
for any license-gated source. Every gated source therefore remains at
`evidence_present_manual_verification_still_required`. Two internal
boundaries (private terminology config boundary, B07 / MKB public-safe
boundary) are `verified_by_prior_boundary_checks` — those are
structural boundary verifications, not legal clearances.

Private adapter implementation and real private-store access remain
**BLOCKED**.

## 2. Why this block exists

`MEDAI-TERM-MESH-LOCAL-DOWNLOAD-PACK-01` (commit `cedbbd3`)
recommended `manual_operator_license_verification_return` as the next
step. This block delivers that return in provisional form: it records
which evidence is present in public-safe reports, marks every gated
source conservatively, and tells the operator exactly what shape the
final explicit return must take. It does not invent verification that
does not exist.

## 3. Resource verification matrix

Allowed status values: `verified_by_explicit_operator_return`,
`evidence_present_manual_verification_still_required`,
`verified_by_prior_boundary_checks`, `not_verified`, `not_applicable`,
`blocked`.

| Resource | Role | Status |
| --- | --- | --- |
| LOINC | primary | **evidence_present_manual_verification_still_required** |
| RxNorm full | primary | **evidence_present_manual_verification_still_required** |
| RxNorm prescribable | auxiliary | **evidence_present_manual_verification_still_required** |
| SNOMED CT US Edition | primary | **evidence_present_manual_verification_still_required** |
| SNOMED CT International | secondary | **evidence_present_manual_verification_still_required** |
| UMLS Metathesaurus | future import (umls_future_gated) | **evidence_present_manual_verification_still_required** |
| MeSH | future integration (helper created, no download) | **evidence_present_manual_verification_still_required** |
| private license-acknowledgement handling | private confirmation | **evidence_present_manual_verification_still_required** |
| private terminology config boundary | internal boundary | **verified_by_prior_boundary_checks** |
| B07 / MKB public-safe boundary | internal boundary | **verified_by_prior_boundary_checks** |

Full per-resource rationale, license class, and evidence class are in
`resource_verification_matrix` in the JSON.

## 4. What evidence IS present

- **Preflight observation** of canonical local presence for LOINC,
  RxNorm (+ prescribable), SNOMED CT US, SNOMED CT International, and
  UMLS via `terminology_sources_preflight`.
- **Synthetic-row imports** for RxNorm + LOINC via `CKA-TERM-02`
  (244,529 + 109,325 = 353,854 concepts) recorded as aggregate counts
  only; no row content in any public report.
- **`.gitignore` protections** for `terminology_data/`,
  `data/terminology/`, `config/terminology_sources.local.json`,
  `LICENSE_ACK_PRIVATE.json`, `**/LICENSE_ACK_PRIVATE*`,
  `**/*TERMINOLOGY_PRIVATE*`, and `scripts/private_local/` confirmed.
- **MeSH helper** for the operator's Windows working copy created by
  `MEDAI-TERM-MESH-LOCAL-DOWNLOAD-PACK-01` targeting the official NLM
  source only; no third-party mirrors; no MeSH bytes downloaded.
- **Private-config boundary** structurally verified by
  `MEDAI-TERM-PRIVATE-CONFIG-VERIFY-02`: ignored by git, never
  committed, never staged, contents never read.
- **B07 / MKB public-safe boundary** validated across `B07 term01
  opt-in`, `CKA-B07 medical coding`, `CKA-B10 preflight scaffold`, and
  `CKA-B11 final MVP release` with feature flags default-off and
  external_api_used=false everywhere.

## 5. What is still NOT legally / operator verified

- **All eight license-gated sources** above lack explicit operator
  return statements confirming license terms have been reviewed and
  accepted for local-only, non-redistributing, no-runtime-external-API
  use.
- License text was never reproduced. License-acknowledgement contents
  were never read.
- Therefore, no source may yet be granted
  `verified_by_explicit_operator_return`.

## 6. MeSH status

`mesh_status` = **`download_helper_created`**.
`mesh_download_completed` = **false**.
`mesh_integration_allowed` = **false**.

The MeSH-download helper at `scripts/private_local/` (gitignored) is
ready for the operator to run on their Windows working copy with
official NLM URLs verified via the canonical download page. No bytes
were downloaded by any block to date. MeSH license terms have not
been operator-returned.

## 7. Private config boundary status

Structurally verified by `TERM-PRIVATE-CONFIG-VERIFY-02`:

- `private_config_ignored_by_git`: true
- `private_config_committed`: false
- `private_config_staged`: false
- `private_config_contents_read` / `printed`: false
- `private_config_path_printed` (full path): false (only the
  gitignore-pattern form appears in any public report)

This is a structural boundary clearance, not a legal license
clearance.

## 8. Why private adapter implementation remains BLOCKED

- `private_adapter_implementation_allowed`: **false**
- `real_private_store_access_allowed`: **false**

Reasons:

1. No explicit operator license return exists for any gated source.
2. MeSH has not yet been downloaded operator-side; only the helper
   exists.
3. No audited private-store adapter has been approved by a SPEC block
   subsequent to `CKA-TERM-INTEGRATION-PRIVATE-ADAPTER-READINESS-02`.
4. The synthetic TERM-05 adapter remains the only approved fixture
   for tests and UAT.
5. The default-off terminology match hypothesis helper continues to
   fail closed when no adapter is injected.

## 9. What the operator must return to unblock implementation

A reports-only operator-return block (recommended id:
`MEDAI-TERM-LICENSE-OPERATOR-RETURN-06`) that records for **each** of
the eight gated sources below:

- `operator_confirms_license_terms_reviewed` (bool)
- `operator_confirms_local_only_use` (bool)
- `operator_confirms_no_redistribution` (bool)
- `operator_confirms_no_runtime_external_api` (bool)
- `operator_confirmation_block_id` (controlled string)

Resources to cover:

1. LOINC
2. RxNorm full
3. RxNorm prescribable
4. SNOMED CT US Edition
5. SNOMED CT International
6. UMLS Metathesaurus
7. MeSH
8. private license-acknowledgement handling

The return statement **must not** contain licensed row content,
license text, or private filesystem paths. Each per-source entry is a
small boolean record plus a controlled-vocabulary status string.

## 10. Safety / privacy confirmation

- `runtime_behavior_changed` / `app_main_modified` / `helper_modified`
  / `streamlit_wiring_changed` / `launcher_files_modified` /
  `startup_preflight_modified` / `config_modified`: **false**.
- `extraction_behavior_changed` / `ocr_behavior_changed` /
  `classifier_behavior_changed` / `threshold_behavior_changed`:
  **false**.
- `cue_expansion_recommended` / `cue_expansion_performed`: **false**.
- `external_api_used_for_runtime` / `external_api_enabled`: **false**.
- `clinical_interpretation_performed` /
  `diagnosis_inference_performed` /
  `medication_inference_performed` /
  `ddi_behavior_changed` /
  `treatment_inference_performed`: **false**.
- `licensed_terminology_rows_read` /
  `licensed_terminology_rows_printed` /
  `licensed_terminology_rows_in_public_reports`: **false**.
- `license_ack_private_read` / `license_ack_private_staged`:
  **false**.
- `private_config_contents_read` / `private_config_staged`: **false**.
- `runtime_db_contents_opened` / `source_documents_opened` /
  `private_files_opened_for_content`: **false**.
- `raw_text_printed` / `raw_filenames_printed` /
  `private_paths_printed` / `secrets_printed`: **false**.
- `tags_created` / `tags_modified` / `prior_park_tags_touched`:
  **false**.
- FREEZE tag pair (`7ef8ffd`) and PARK-20..23 tag pairs unchanged on
  origin.

## 11. Recommended next step

**`MEDAI-TERM-LICENSE-OPERATOR-RETURN-06`** — a reports-only block
that records the operator's explicit per-resource license-review
confirmations using only boolean fields and controlled-vocabulary
status strings. Once that return is received and recorded,
`private_adapter_implementation_allowed` may be reconsidered by a
subsequent SPEC block. Until then, private adapter implementation
remains blocked.

Cue expansion remains explicitly **NOT** recommended.
