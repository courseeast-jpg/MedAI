# MEDAI-TERM-MESH-LOCAL-DOWNLOAD-PACK-01 — Local-Only MeSH Download Pack and Private Config Update

Local-only private data handling + public-safe reports block. No
runtime code changes. No adapter implementation. No terminology
import. No licensed row access. No runtime DB access. No external APIs
for runtime. No tags created. PARK-20..23 tag pairs and the FREEZE tag
pair remain intact. Cue expansion remains explicitly **NOT**
recommended.

## 1. Executive summary

- A gitignored MeSH-download helper for the operator's Windows working
  copy was created at
  `scripts/private_local/download_mesh_2026_local.ps1`.
- A one-block `.gitignore` protection for `scripts/private_local/` was
  added (this is the only tracked change in the implementation commit
  besides public-safe reports).
- The helper targets the **official NLM source only**
  (`https://www.nlm.nih.gov/databases/download/mesh.html`) for current
  production-year MeSH XML files. **No third-party mirrors are used.**
- No MeSH bytes were transferred by this block. Direct outbound access
  from this public Linux mirror to `www.nlm.nih.gov` returned HTTP 403
  `host_not_allowed` — the operator must run the helper on their
  Windows working copy where the private terminology storage lives.
- `mesh_dataset_status` = **`download_helper_created`**.
- Private config was **not** updated by this block; if the operator
  successfully runs the helper, they may then privately edit the
  config on their machine to set MeSH status to
  `downloaded_local_private`.
- Manual license verification remains **`still_required`**.
- Private adapter implementation and real private-store access remain
  **BLOCKED**.

## 2. Why this block exists after VERIFY-02

`TERM-PRIVATE-CONFIG-VERIFY-02` (commit `60f1114`) confirmed the
private terminology config boundary is correct and that
`mesh_status = manual_download_required` carried forward from
`TERM-PRIVATE-CONFIG-01`. The recommended next step was a
manual-license-verification-or-MeSH-download-pack. This block delivers
the MeSH-download-pack half: a safe, official-source-only, local-only
acquisition helper plus public-safe reports — without performing any
download from this public mirror.

## 3. Official NLM source used

| Field | Value |
| --- | --- |
| Canonical download page | `https://www.nlm.nih.gov/databases/download/mesh.html` |
| Canonical URL pattern targeted | `https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/{desc\|qual\|supp}{YYYY}.xml` |
| Third-party mirrors used | **false** |
| Operator must verify URLs against canonical page before run | **true** (helper refuses unless `MEDAI_MESH_NLM_URLS_VERIFIED=1`) |
| Production year targeted | `2026` |

## 4. MeSH acquisition result

`mesh_dataset_status` = **`download_helper_created`**.

| Signal | Value |
| --- | :-: |
| `mesh_download_attempted` | false |
| `mesh_files_staged` | false |
| `mesh_files_committed` | false |
| `mesh_contents_read` | false |
| Helper relative path | `scripts/private_local/download_mesh_2026_local.ps1` |
| Helper is gitignored | true |
| Helper was staged | false |
| Helper was committed | false |
| Target relative dest dir | `terminology_data/mesh/2026` (gitignored under `terminology_data/`) |
| Target XML files planned | `desc2026.xml`, `qual2026.xml`, `supp2026.xml` |
| Local-only README created by helper on success | true |

Reason downloads were not performed in this block: direct outbound
access to `www.nlm.nih.gov` from this public Linux mirror is blocked
by the sandbox firewall (HTTP 403 `host_not_allowed`). Per the task,
this falls cleanly under the documented `download_helper_created`
outcome; the operator must execute the helper on their Windows
working copy.

## 5. Private config update result

The block did **not** edit `config/terminology_sources.local.json`.

| Signal | Value |
| --- | :-: |
| `private_config_updated_by_this_block_on_operator_machine` | false |
| `private_config_contents_printed` | false |
| `private_config_path_printed` | false |
| `private_config_staged` | false |
| `config_committed` | false |

Operator-side action recommended after a successful run of the helper
on Windows: privately edit the config to set the MeSH source status
to `downloaded_local_private`, recording only the relative path used
by the helper. Must not print private contents or absolute paths in
any committed file.

## 6. What was NOT read

- `config/terminology_sources.local.json` contents — never opened.
- `LICENSE_ACK_PRIVATE.json` contents — never opened.
- Licensed terminology rows (LOINC / RxNorm / SNOMED / UMLS / MeSH) —
  never read.
- `terminology_data/` row contents — never read.
- `data/terminology/` row contents — never read.
- Runtime DB rows — never inspected.
- Source documents — never opened.
- Raw OCR text, raw document text — never read.
- Keys, secrets, tokens — never read.

## 7. What was NOT staged

- `config/terminology_sources.local.json` — not staged, not committed.
- Anything under `terminology_data/` (including any future MeSH
  downloads) — not staged, not committed.
- Anything under `data/terminology/` — not staged.
- `LICENSE_ACK_PRIVATE.json` — not staged.
- `scripts/private_local/download_mesh_2026_local.ps1` — gitignored,
  not staged, not committed.
- Any future MeSH XML files — not stageable (covered by both
  `terminology_data/` and the `*.RRF` / archive ignore patterns).
- Private scripts, private documents, runtime DBs, backups, bundles,
  keys, secrets — not staged.

Staged into the implementation commit: only the public-safe reports
under `reports/medai_term_mesh_local_download_pack_01/` and the one
additive `.gitignore` block adding protection for
`scripts/private_local/`.

## 8. Manual license verification still required

`manual_license_verification_status` = **`still_required`**.

Adding MeSH to the integration scope does not relax any of the prior
license-verification gates (LOINC / RxNorm / SNOMED CT US / SNOMED CT
International / UMLS). The operator's manual clearance — covering
MeSH terms as well now — remains the canonical path to unblock any
adapter implementation.

## 9. Why private adapter implementation remains BLOCKED

- `private_adapter_implementation_allowed`: **false**
- `real_private_store_access_allowed`: **false**

Reasons:

1. Manual license verification has not cleared for the full source
   set (now including MeSH).
2. MeSH itself has not yet been downloaded operator-side; this block
   only created the helper.
3. No audited private-store adapter has been approved by a SPEC block
   subsequent to `CKA-TERM-INTEGRATION-PRIVATE-ADAPTER-READINESS-02`.
4. The synthetic TERM-05 adapter remains the only approved fixture
   for tests and UAT.
5. The default-off terminology match hypothesis helper continues to
   fail closed when no adapter is injected.

## 10. Safety / privacy confirmation

- `runtime_behavior_changed` / `app_main_modified` / `helper_modified` /
  `streamlit_wiring_changed` / `launcher_files_modified` /
  `startup_preflight_modified` / `config_committed`: **false**.
- `extraction_behavior_changed` / `ocr_behavior_changed` /
  `classifier_behavior_changed` / `threshold_behavior_changed`:
  **false**.
- `cue_expansion_recommended` / `cue_expansion_performed`: **false**.
- `external_api_used_for_runtime` / `external_api_enabled`: **false**.
  (Note: the helper itself is not runtime code; it is an operator
  Windows-side acquisition script, gitignored, and would call NLM only
  during operator-side execution.)
- `clinical_value_parsing_performed` / `clinical_interpretation_performed`
  / `diagnosis_inference_performed` / `medication_inference_performed`
  / `ddi_behavior_changed` / `treatment_inference_performed`
  / `abbreviation_expansion_performed`: **false**.
- `licensed_terminology_rows_read` / `licensed_terminology_rows_printed`:
  **false**.
- `license_ack_private_read`: **false**.
- `runtime_db_contents_opened` / `source_documents_opened` /
  `private_files_opened_for_content`: **false**.
- `raw_text_printed` / `raw_filenames_printed` /
  `private_paths_printed` / `secrets_printed`: **false**.
- `tags_created` / `tags_modified` / `prior_park_tags_touched`:
  **false**.
- FREEZE tag pair (`7ef8ffd`) and PARK-20..23 tag pairs unchanged on
  origin.

## 11. Recommended next step

`manual_operator_license_verification_return` — a reports-only block
that packages the operator's manual license clearances (now including
MeSH terms in addition to LOINC / RxNorm / SNOMED / UMLS) into a
public-safe verification receipt without reading licensed rows.
Private adapter implementation remains blocked until those gates
clear.

Cue expansion remains explicitly **NOT** recommended.
