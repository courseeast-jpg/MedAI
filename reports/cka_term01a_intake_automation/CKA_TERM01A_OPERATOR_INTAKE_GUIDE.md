# CKA-TERM-01A Operator Intake Guide

This guide automates the local **operator intake** workflow for
licensed terminology files (UMLS, SNOMED CT, RxNorm, LOINC). TERM-01A
**does not download** anything, **does not bypass licensing**, and
**does not commit** terminology data.

The actual licensed-import is deferred to **CKA-TERM-02 controlled
local terminology import**, which is opened only after the operator
has placed real licensed files locally and explicitly acknowledged
their license.

---

## Intake workflow at a glance

```
1.  python scripts/cka_terminology_prepare_intake.py
       -> creates terminology_data/{loinc,rxnorm,umls,snomed_ct}/
       -> drops terminology_data/LICENSE_ACK_PRIVATE.template.json
       -> never creates the real ack file

2.  Operator downloads licensed files from each vendor's portal
    (UMLS Metathesaurus, SNOMED CT, RxNorm, LOINC) and copies them
    into the appropriate terminology_data/<system>/ folder.

3.  Operator copies the template:
       cp terminology_data/LICENSE_ACK_PRIVATE.template.json \
          terminology_data/LICENSE_ACK_PRIVATE.json
    Operator edits the new file:
       {"operator_acknowledged": true,
        "acknowledged_systems": ["umls", "snomed_ct", "rxnorm", "loinc"]}
    Both .template.json and .json are gitignored — neither is staged.

4.  python scripts/cka_terminology_check_ready.py
       -> reports inventory state per system
       -> tells the operator exactly which systems still need ack

5.  When all systems are import_ready, open CKA-TERM-02 (separate
    block, separate operator approval).
```

---

## Hard rules

- **No network**, no vendor API, no automated download.
- **No fake acknowledgments.** The template ships with
  `operator_acknowledged=false` and an empty `acknowledged_systems` list.
  The operator must produce the real ack file by hand.
- **No raw paths** in any public report — only safe hashes and counts.
- **No clinical text** generated. No prescribing. No diagnosis.
- **No B07 default change.** The medical-coding default behaviour
  (`synthetic_mapper`) is unchanged.
- **No DDI status modification.** Coding does NOT clear DDI status.
- **No hypothesis promotion.** Coding does NOT promote AI-derived
  facts to the active tier.

---

## Filename classification

The classifier inspects **only filenames**, never contents:

| Pattern fragment (case-insensitive) | System |
|---|---|
| `Loinc.csv`, `LoincTable.csv`, `Loinc*.zip`, `loinc_` | LOINC |
| `RXNCONSO.RRF`, `RXNREL.RRF`, `RXNSAT.RRF`, `RxNorm*.zip` | RxNorm |
| `MRCONSO.RRF`, `MRSTY.RRF`, `MRREL.RRF`, `umls*.zip`, `mmsys*.zip` | UMLS |
| `sct2_Concept*`, `sct2_Description*`, `sct2_Relationship*`, `SnomedCT*.zip` | SNOMED CT |
| anything else | **unknown** |

ZIP archives are classified but **not extracted by default**. Use
`--extract-approved` only when the operator has confirmed local
license rights and accepts the zip-slip-protected extraction.

---

## Optional scan / copy / extract flags

```
python scripts/cka_terminology_prepare_intake.py \
    [--scan <folder>] \
    [--recurse] \
    [--copy-approved] \
    [--extract-approved]
```

- `--scan <folder>` — opt-in local scan of a folder. Off by default.
  Reads filenames only; never opens contents.
- `--recurse` — recurse into the scan folder. Default off.
- `--copy-approved` — copy classified files into
  `terminology_data/<system>/`. Refused without `--scan`.
  **Refuses to write outside the resolved `terminology_data/` tree.**
- `--extract-approved` — extract classified ZIP archives into
  `terminology_data/<system>/`. **Zip-slip protection** is always on:
  any entry whose path resolves outside the system subdir is blocked
  and counted under `entries_blocked_zip_slip`.

None of these flags upload, download, or call any vendor API.

---

## Readiness checker

```
python scripts/cka_terminology_check_ready.py
```

Prints a JSON block with:

- `inventory` — TERM-01 inventory of `terminology_data/`.
- `readiness.systems_present` — systems with files on disk.
- `readiness.systems_acknowledged` — systems present **AND** in the
  operator's `LICENSE_ACK_PRIVATE.json`'s `acknowledged_systems`.
- `readiness.systems_import_ready` — systems present **AND**
  acknowledged — TERM-02 can import these.
- `readiness.systems_license_required` — systems present but NOT
  acknowledged.
- `readiness.systems_missing` — systems with no files.
- `readiness.pending_acknowledgments` — systems the operator still
  needs to add to the acknowledgment file.
- `operator_action_required` — human-readable next step.

If `pending_acknowledgments` is non-empty, the operator must edit
their `terminology_data/LICENSE_ACK_PRIVATE.json` to include those
systems before TERM-02 can run.

---

## Boundaries (binding under TERM-01A)

| Boundary | Value |
|---|---|
| `folders_prepared` | true |
| `ack_template_ready` | true |
| `real_ack_created` | **false** |
| `file_classifier_ready` | true |
| `zip_slip_protection_ready` | true |
| `inventory_runner_ready` | true |
| `local_scan_default_off` | **true** |
| `real_terminology_downloaded` | **false** |
| `real_terminology_imported` | **false** |
| `real_terminology_files_committed` | **false** |
| `license_gate_bypassed` | **false** |
| `external_api_used` | **false** |
| `external_terminology_api_used` | **false** |
| `license_text_written_to_public_reports` | **false** |
| `clinical_recommendations_generated` | **false** |
| `prescription_dosing_advice_generated` | **false** |
| `production_ocr_changed` | **false** |
| `production_extractor_changed` | **false** |
| `safety_gate_changed` | **false** |
| `frozen_hitl_release_reopened` | **false** |

The next manual action is for the operator to download licensed
files and create their private license acknowledgment. The next code
action is **CKA-TERM-02 controlled local terminology import**.
