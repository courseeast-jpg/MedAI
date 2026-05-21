# MEDAI-REAL-DOC-B4-PATHOLOGY-FAMILY-EVAL-01 — Evaluation-Only Pathology Report Family Audit

Evaluation-only audit triggered by a real operator signal. The operator
observed a real local document returning Unknown / rejected through the
frozen MedAI pipeline and visually identified the document as a
pathology / dermatopathology consultation report. The triage block
(`MEDAI-REAL-DOC-UNKNOWN-TRIAGE-01`) matched the failure to bucket
**B4 — unsupported or under-covered document family**. This block
audits whether MedAI should add a future safe document-family
classifier branch for pathology / dermatopathology reports. **No**
classifier change. **No** cue expansion. **No** threshold change. **No**
OCR routing change. **No** parser change. **No** UI / Streamlit /
launcher / startup / config change. **No** persistence change. **No**
external API call. **No** real PDF, screenshot, diagnosis, raw text,
raw OCR text, raw filename, private path, PHI, or runtime DB row read
or printed. V1 frozen release at `7ef8ffd` remains preserved. Cue
expansion remains explicitly **NOT** recommended.

## A. Scope And Non-Scope

**In scope**

- Inspect the existing classifier and family-registry source.
- Determine whether the current classifier supports pathology reports.
- Catalog the safe structural cues that would later define the family.
- Recommend a controlled family name and a controlled subtype
  vocabulary.
- Catalog the future extraction fields that would be useful.
- Decide whether a future implementation block is justified.
- Recommend the smallest safe next block.

**Out of scope**

- Reading the operator's real PDF, screenshot, diagnosis, raw text, raw
  OCR text, raw filename, or private paths.
- Adding any classifier cue.
- Changing any classifier rule.
- Changing any threshold.
- Changing OCR routing.
- Changing the parser.
- Changing the UI, launchers, startup, or config.
- Changing persistence.
- Calling an external API.
- Creating, moving, or deleting tags.
- Implementing the future spec.

## B. Trigger And Bucket Mapping

- Operator-observed failure: real local pathology / dermatopathology
  consultation report returned Unknown / rejected.
- Operator-observed structural sections (public-safe summary): specimen,
  signs and symptoms, clinical diagnosis, gross description,
  microscopic description, diagnosis, ICD code field, comments / margin,
  photomicrograph.
- Triage matched bucket: **B4 unsupported_or_under_covered_document_family**.
- Bucket B4 permits an evaluation-only follow-up; it must not implement
  any classifier or cue change.

## C. Core Classifier Inspection (Read-Only)

Source: `document_classification/document_classifier.py`. Entry point:
`classify_document`.

Vocabulary (the `DOCUMENT_TYPES` constant) lists seven labels:
`clinical_note`, `imaging_report`, `lab_report`,
`microbiology_pcr_report`, `prescription`, `unknown_medical`,
`unknown_nonmedical`. Only four of those seven are actually emitted by
the live rules in `classify_document`: `lab_report` (Rule 1),
`prescription` (Rule 2 and Rule 4a), `microbiology_pcr_report` (Rule 3
and Rule 4b), `unknown_medical` (Rule 4c and Rule 5).

| Check | Result |
| --- | :-: |
| Has any pathology-specific token in `_LAB_TOKENS_EN`? | **no** |
| Has any pathology-specific token in `_MICROBIOLOGY_TOKENS_EN`? | **no** |
| Has any pathology-specific token in `_PRESCRIPTION_TOKENS_EN`? | **no** |
| Has "dermatopathology" cue? | **no** |
| Has "biopsy" cue? | **no** |
| Has "microscopic" cue? | **no** |
| Has "photomicrograph" cue? | **no** |
| Has "gross description" cue? | **no** |
| Lists `specimen` as a strong lab token? | **no** |
| Lists `specimen` as a weak `_LAB_INDICATOR_RE` only? | **yes** |

**Synthetic call** (against the generic synthetic text "Specimen
received. Clinical diagnosis: review pending. Gross description.
Microscopic description. Final diagnosis. ICD code field.", no real
document text):

| Field | Synthetic result |
| --- | --- |
| `document_type` | `unknown_medical` |
| `confidence` | 0.30 |
| `review_reason` | `unknown_document_type` |
| `evidence` | `no_strong_signals` |
| `warning` | `low_confidence_document_type` |

This confirms the core classifier emits `unknown_medical` on a
generic pathology-shaped synthetic payload. Rule 5 (safe default)
fires because no lab / prescription / microbiology cue and no
Cyrillic-dominant fallback applies.

## D. Metadata Family Registry Inspection (Read-Only)

Source: `app/document_type_registry.py`. Entry point:
`document_family_classification_diagnostic`.

The registry already declares a `PATHOLOGY_REPORT_LABEL` family rule
(`"Pathology report"`) with:

| Attribute | Value |
| --- | --- |
| Threshold | 2 cue keys |
| Required-any keys | none |
| English cue keys | `specimen_section`, `microscopic_description_section`, `pathology_conclusion_section` |
| Russian cue keys | same three keys, with Cyrillic surface terms |
| Polish cue keys | same three keys, with Polish surface terms |
| Albanian cue keys | same three keys, with Albanian surface terms |
| Language pack coverage | 4 packs |
| Review-only default | **true** |
| Auto-accept allowed | **false** |

**Synthetic call** on the same generic synthetic text:

| Field | Synthetic result |
| --- | --- |
| `candidate_family` | `Pathology report` |
| `matched_family_cue_keys` | `specimen_section`, `microscopic_description_section`, `pathology_conclusion_section` (3 of 3) |
| `classification_block_reason` | `classified` |
| `review_only` | **true** |
| `auto_accept_allowed` | **false** |

## E. Two-Layer Split Summary

| Layer | Field emitted | Synthetic value for pathology |
| --- | --- | --- |
| Core `document_classifier.py` | `document_type` | `unknown_medical` |
| Metadata `document_type_registry.py` | `document_family_classification_diagnostic.candidate_family` | `Pathology report` |
| UI primary card (`operator_document_type(item)`) | reads `document_type` | shows `Unknown` |
| UI secondary family badge (DIAG-08A wiring) | reads `document_family_classification_diagnostic` | shows `Pathology report` (when present) |
| Auto-accept | gated by safety stack | **false** |
| Review-bound | preserved | **true** |

Conclusion: pathology has **partial coverage today** at the metadata
family layer. The operator-visible primary "Document type" label still
shows Unknown because the core classifier (which produces that label)
has no pathology rule. The secondary family-diagnostic badge can
already show "Pathology report" when the cue threshold is met, but the
operator may not be looking at that badge, or the threshold may not
have been met because the operator's real document uses section
headings (for example "gross description", "photomicrograph", "ICD
code field") that are not yet in the current 3-cue rule.

## F. Safe Structural Cue Inventory (Future Spec Only)

Cues already in the family rule (must not be added or expanded here):

- english: `specimen`, `microscopic description`, `pathology diagnosis`,
  `final diagnosis`.
- russian / polish / albanian: parallel translations for the same three
  cue keys.

Cues observed in the operator's structural description but **not yet
covered** by the family rule (candidate additions for a future spec
block, not added in this block):

- `gross description`
- `clinical diagnosis`
- `signs and symptoms`
- `reason for biopsy`
- `photomicrograph`
- `margins`
- `comments`
- `icd code` (structural marker; the code value itself remains
  review-only and would still pass through the privacy scanner)

Any future spec block must:

- treat these as planning-level only;
- avoid copying the operator's real text;
- avoid raising the auto-accept ceiling;
- preserve the review-bound default;
- pass through the V2 default-off status registry as a `spec_only` or
  `implemented_default_off` entry.

## G. Controlled Family Name Recommendation

| Choice | Outcome |
| --- | --- |
| Recommended controlled family name | **`pathology_report`** |
| Reason | matches the existing `PATHOLOGY_REPORT_LABEL` constant and the existing metadata family rule; avoids fragmenting the registry |
| Alternative `dermatopathology_report` | rejected as a top-level family (would multiply registry); preferred as a subtype |
| Alternative `biopsy_pathology_report` | rejected as a top-level family; preferred as a subtype |
| Alternative `histopathology_report` | rejected as a top-level family; preferred as a subtype |

Recommended subtype field name: `pathology_subtype`. Recommended
controlled vocabulary (planning-level only, not implemented):

- `dermatopathology`
- `biopsy`
- `cytology`
- `frozen_section`
- `consultation_pathology`
- `unspecified_pathology`

## H. Future Extraction Field Inventory

| # | Field | Operator observed | Current metadata cue covers it? | Future extraction useful? |
| -: | --- | :-: | :-: | :-: |
| 1 | `specimen` | yes | yes | yes |
| 2 | `clinical_diagnosis_or_reason_for_biopsy` | yes | no | yes |
| 3 | `gross_description` | yes | no | yes |
| 4 | `microscopic_description` | yes | yes | yes |
| 5 | `final_diagnosis` | yes | yes | yes |
| 6 | `icd_code` | yes | no | yes |
| 7 | `margins_or_comments` | yes | no | yes |
| 8 | `image_or_photomicrograph_presence` | yes | no | yes |
| 9 | `signs_and_symptoms` | yes | no | yes |

Every field listed is structural metadata only. Actual values are
review-bound and **must not** be auto-accepted. ICD code values stay
inside the privacy scanner envelope (no PHI; ICD codes themselves are
controlled-vocabulary identifiers, but their associated free-text
descriptions remain review-only).

## I. Future Implementation Decision

| Question | Answer |
| --- | --- |
| Does the current core classifier support pathology / dermatopathology? | **no** (emits `unknown_medical`) |
| Does the metadata family registry support pathology? | **partial** (3-cue rule, threshold 2, 4 language packs) |
| Are the operator's section headings adequately covered? | **partial** (3 of 9 covered) |
| Is a future implementation justified? | **yes**, as a reports-only / synthetic-only spec block, not a behavior change |
| Should the future block expand classifier cues? | **no, not in the spec block**; cue expansion remains blocked by the V2 default-off status registry |
| Should the future block touch auto-accept? | **no** |
| Should the future block touch OCR routing? | **no** |
| Should the future block read the operator's real document? | **no** |

## J. Recommended Smallest Safe Next Block

**Primary candidate:**
`MEDAI-PATHOLOGY-REPORT-FAMILY-SPEC-01` — reports-only / synthetic-only
pathology family spec. The spec block must:

- inspect the existing family rule;
- catalog the candidate cue additions at planning level only;
- catalog the future extraction fields at planning level only;
- declare a controlled `pathology_subtype` vocabulary at planning level
  only;
- emit three public-safe reports + audit script + focused tests;
- preserve every safety / privacy invariant from this evaluation block;
- not add cues to `document_classifier.py`;
- not change `document_type_registry.py` cue groups;
- not change thresholds, OCR routing, parsers, persistence, UI,
  launchers, startup, or config;
- avoid reading the operator's real document or any private artifact.

**Fallback:** `FREEZE-MAINTENANCE-ONLY` if the operator chooses not to
draft the spec yet.

## K. Why No Behavior Is Changed In This Block

- freeze-maintenance posture forbids classifier behavior change without
  explicit approval;
- bucket B4 triage permits evaluation-only follow-up, not
  implementation;
- cue expansion is explicitly blocked by the V2 default-off status
  registry (entry: `cue_expansion`, status: `blocked`);
- any classifier rule change must go through a separate spec block and
  a privacy / safety review;
- the operator has not yet approved a behavior change; this block
  intentionally surfaces evidence only.

## L. Safety / Privacy Invariants

| Invariant | Value |
| --- | :-: |
| `block_mode` | `evaluation_only_pathology_family_audit` |
| `evaluation_only` | **true** |
| `synthetic_only` | **true** |
| `triggered_by_real_operator_signal` | **true** |
| `matched_unknown_triage_bucket` | `B4` |
| `operator_supplied_real_document_artifact` | **false** |
| `real_document_inspected_by_assistant` | **false** |
| `real_pdf_committed` | **false** |
| `real_screenshot_committed` | **false** |
| `real_diagnosis_printed` | **false** |
| `raw_text_read` / `raw_text_printed` | **false** |
| `raw_ocr_text_read` / `raw_ocr_text_printed` | **false** |
| `raw_filenames_read` / `raw_filenames_printed` | **false** |
| `private_paths_printed` | **false** |
| `secrets_printed` | **false** |
| `phi_inspected` | **false** |
| `implementation_started` | **false** |
| `new_helper_created` | **false** |
| `direct_implementation_recommended` | **false** |
| `behavior_changed_in_this_block` | **false** |
| `runtime_behavior_changed` | **false** |
| `app_main_changed` | **false** |
| `streamlit_code_changed` | **false** |
| `ui_changed` | **false** |
| `launcher_changed` | **false** |
| `installer_changed` | **false** |
| `deployment_script_changed` | **false** |
| `startup_config_changed` | **false** |
| `extraction_changed` | **false** |
| `ocr_changed` | **false** |
| `ocr_routing_changed` | **false** |
| `classifier_changed` | **false** |
| `classifier_cue_added` | **false** |
| `classifier_rule_changed` | **false** |
| `threshold_scoring_changed` | **false** |
| `parser_behavior_changed` | **false** |
| `fallback_behavior_changed` | **false** |
| `cue_pack_changed` | **false** |
| `cue_expansion_recommended` | **false** |
| `db_schema_changed` | **false** |
| `migration_created` / `migration_executed` | **false** |
| `persistence_code_changed` | **false** |
| `clinical_behavior_changed` / `ddi_behavior_changed` / `terminology_behavior_changed` | **false** |
| `private_adapter_implemented` / `concrete_adapters_implemented` / `runtime_wiring_added` | **false** |
| `external_api_used` | **false** |
| `private_data_accessed` / `licensed_rows_read` / `licensed_rows_exposed` / `private_license_ack_read` / `private_config_read` / `source_documents_opened` | **false** |
| `runtime_db_accessed` | **false** |
| `tags_touched` / `tags_created` / `tags_modified` | **false** |
| `v1_release_preserved` | **true** |
| `local_only_default` | **true** |
| `review_bound_default` | **true** |
| `external_api_blocked_default` | **true** |
| `auto_accept_allowed_default` | **false** |

## M. Recommended Next 3-Block Sequence

1. `MEDAI-PATHOLOGY-REPORT-FAMILY-SPEC-01_OR_FREEZE-MAINTENANCE-ONLY`
   (primary).
2. `FREEZE-MAINTENANCE-ONLY_IF_OPERATOR_DEFERS_THE_SPEC`.
3. `V2-ROADMAP-04_IF_NEW_SIGNAL_ELSE_FREEZE-MAINTENANCE-ONLY`.

Forbidden in any follow-up block:

- adding pathology cues to `document_classifier.py`;
- changing cue groups in `document_type_registry.py`;
- changing OCR routing, thresholds, parsers, or extractors;
- reopening cue expansion (explicitly **NOT** recommended);
- reopening terminology / private adapter implementation;
- touching any existing parking / freeze tag;
- committing the operator's real PDF, real screenshot, real diagnosis,
  raw text, raw OCR text, raw filenames, private paths, PHI, or
  runtime DB rows.

V1 frozen release at `7ef8ffd` continues to be the durable shipped
artifact. Freeze-maintenance posture remains intact.
