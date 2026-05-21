# MEDAI-REAL-DOC-UNKNOWN-TRIAGE-01 — Local Real-Document Unknown Result Triage

Diagnosis-only triage block. The operator observed a real local document
returning Unknown / rejected / no useful result through the frozen MedAI
pipeline. This block defines a public-safe diagnostic framework that lets
the operator map the failing run to one of eight failure buckets using
only the existing public-safe diagnostic surface that the UI already
exposes. **No** runtime behavior change. **No** classifier change. **No**
OCR routing change. **No** threshold or cue change. **No** persistence
change. **No** external API call. V1 frozen release at `7ef8ffd` remains
preserved. PARK-20..23 / PARK-24..26 / FREEZE / TERM helper-wiring
PARK-01 / license-gate PARK-02 tag groups remain intact. Cue expansion
remains explicitly **NOT** recommended.

## A. Scope And Non-Scope

**In scope**

- Define a public-safe taxonomy of eight Unknown / rejected failure
  buckets the operator may observe locally.
- Catalog the 20 public-safe advanced diagnostic fields already exposed
  by the run-result card.
- Provide a decision tree that maps observable diagnostic signals to a
  single failure bucket.
- Provide a bucket-conditional smallest next block recommendation
  table.
- Preserve the freeze-maintenance posture: no implementation in this
  block.

**Out of scope**

- Reading the operator's real PDF.
- Reading any raw extracted text or raw OCR text.
- Printing raw filenames or private paths.
- Printing PHI, secrets, licensed terminology rows, or private config
  contents.
- Changing the classifier, OCR routing, thresholds, cue packs, parsers,
  or extractors.
- Changing the UI, launchers, startup, or config.
- Calling an external API.
- Creating, moving, or deleting tags.
- Implementing a fix.

## B. Trigger And Operator-Observed Signal

The operator observed a real local document returning Unknown / rejected
/ no useful result. Under freeze-maintenance doctrine this signal
explicitly permits a diagnosis-only triage block. It does not permit
behavior change, cue expansion, OCR routing change, or threshold
adjustment.

The triage is intentionally scoped to public-safe diagnostics only. The
assistant has not inspected the real PDF, raw text, raw OCR text, raw
filenames, private paths, PHI, or runtime DB rows in this block.

## C. Public-Safe Diagnostic Surface

The frozen run-result card already exposes a 20-field advanced
diagnostic panel. These fields are public-safe (no PHI, no raw text, no
raw filename, no private path). The operator reads them locally without
copying any raw content into a public report.

| # | Field | Triage value |
| -: | --- | --- |
|  1 | `document_type` | classifier label assigned to the document |
|  2 | `confidence` | classifier confidence (0.30 floor for the safe default unknown_medical) |
|  3 | `validation_status` | downstream validation outcome (for example, empty, review_ocr_quality) |
|  4 | `selected_extractor` | which extractor produced the upstream payload |
|  5 | `ocr_quality_band` | OCR quality category if OCR ran |
|  6 | `language_text_visibility` | observed script / language visibility |
|  7 | `cyrillic_ocr_recommended` | whether language-aware OCR was recommended |
|  8 | `ocr_gate_reason` | reason the OCR gate produced its outcome |
|  9 | `ocr_gate_fallback_executed` | whether the OCR fallback engine ran |
| 10 | `ocr_gate_fallback_engine` | which OCR fallback engine was used |
| 11 | `ocr_gate_fallback_language` | language used by the OCR fallback |
| 12 | `ocr_gate_fallback_cyrillic_detected` | Cyrillic signal after fallback |
| 13 | `ocr_gate_fallback_text_visibility` | text visibility after fallback |
| 14 | `ocr_gate_fallback_review_only` | whether the fallback result was review-only |
| 15 | `ocr_gate_fallback_auto_accept_allowed` | auto-accept stance after fallback |
| 16 | `ocr_gate_fallback_classification_diagnostic` | post-fallback classification diagnostic |
| 17 | `ocr_gate_fallback_treatment_classification_diagnostic` | post-fallback treatment-class diagnostic |
| 18 | `document_family_classification_diagnostic` | family-level classification diagnostic |
| 19 | `operator_review_reason` | review reason code surfaced to the operator |
| 20 | `operator_reason_label` | aggregate label for the reason surface |

All twenty fields are already wired through the parked DOC-TYPE-UNKNOWN
diagnostic chain (DIAG-01 through DIAG-21) and remain available without
any code change.

## D. Failure Bucket Taxonomy (8)

| ID | Bucket | One-line summary |
| :-: | --- | --- |
| B1 | `file_not_queued_or_run_not_executed` | the file never reached the pipeline (workflow signal, not a classifier outcome) |
| B2 | `text_extracted_but_classifier_cues_insufficient` | extractor returned text but no classifier rule fired; safe default unknown_medical assigned |
| B3 | `ocr_fallback_needed_but_not_triggered` | low OCR quality but the gate did not escalate to OCR |
| B4 | `document_family_unsupported` | legitimate clinical record outside the supported classifier family coverage |
| B5 | `image_or_table_or_layout_heavy_document` | dominated by images, scanned tables, or complex layout |
| B6 | `language_or_script_mismatch` | language or script not adequately covered for this family |
| B7 | `parser_unsupported` | parser layer rejected the upstream payload (unsupported file type, encrypted PDF, malformed) |
| B8 | `ui_run_state_confusion` | operator interpreted a stale or unrelated run as the failing result |

Each bucket's full signal pattern is encoded in section
`failure_bucket_taxonomy` of the JSON report, with primary and secondary
signals enumerated.

## E. Decision Tree (Signal -> Bucket)

The operator runs the tree locally against the run-result card. The
tree is intentionally shallow (five steps) and uses only the 20 fields
from section C.

1. **Is there any run card for the file at all?** Active run results
   contain a matching entry.
   - **no** -> B1 or B8.
   - **yes** -> step 2.
2. **Is `validation_status` equal to `empty`?**
   - **yes** -> B7 if `selected_extractor` reports a parser failure;
     else B3 or B5.
   - **no** -> step 3.
3. **Is `validation_status` equal to `review_ocr_quality`, or is
   `ocr_quality_band` low?**
   - **yes** -> B3 if `ocr_gate_fallback_executed` is False; else B5.
   - **no** -> step 4.
4. **Is `cyrillic_ocr_recommended` True, or does
   `language_text_visibility` indicate non-English / Cyrillic / mixed?**
   - **yes** -> B6.
   - **no** -> step 5.
5. **Did the extractor return text yet `document_type` resolves to
   `unknown_medical` with `confidence` approximately 0.30 and
   `operator_review_reason` equal to `unknown_document_type`?**
   - **yes** -> B2 if cues should plausibly cover this family; else B4.
   - **no** -> review individual fields, re-route via secondary signals,
     and if none fit record the anomaly for a future triage block.

## F. Bucket-Conditional Smallest Next Block

Every entry below is an **evaluation block only**. None of them changes
classifier, OCR, threshold, cue, parser, persistence, UI, or external
behavior. Each is a candidate; none is committed to here.

| Bucket | Smallest next block candidate | Code fix justified? |
| :-: | --- | :-: |
| B1 | none; operator re-runs the file and confirms run card appears | no |
| B2 | `MEDAI-CLASSIFIER-CUE-COVERAGE-EVAL-01` (eval-only) | conditional |
| B3 | `MEDAI-OCR-GATE-MISS-EVAL-01` (eval-only) | conditional |
| B4 | `MEDAI-DOC-FAMILY-COVERAGE-EVAL-01` (eval-only) | conditional |
| B5 | `MEDAI-LAYOUT-HEAVY-EVAL-01` (eval-only) | no |
| B6 | `MEDAI-LANGUAGE-SCRIPT-COVERAGE-EVAL-01` (eval-only) | conditional |
| B7 | `MEDAI-PARSER-COVERAGE-EVAL-01` (eval-only) | conditional |
| B8 | `MEDAI-UI-RUN-STATE-CLARITY-EVAL-01` (read-only UI legend audit) | no |

Conditional means: an evaluation block may be approved only after the
operator records the matched bucket id and explicitly requests the
follow-up. Cue expansion remains explicitly **NOT** recommended for any
bucket.

## G. Was The File Actually Queued And Processed?

This block cannot answer that for the operator's specific document
because the assistant did not inspect the operator's runtime DB, did not
read raw filenames, and did not access private paths. The operator
answers locally using step 1 of section E (decision tree). If step 1
returns **no**, the file is bucket B1 (workflow) or B8 (UI state) and no
classifier or extractor pathway is involved.

## H. Why Did It Become Unknown?

Two structural causes produce Unknown / rejected in the frozen
classifier:

- The safe-default branch (Rule 5 in `document_classifier.py`) assigns
  `document_type = "unknown_medical"`, `confidence = 0.30`, and
  `review_reason = "unknown_document_type"` whenever the typed cue rules
  (lab, prescription, microbiology) and the Cyrillic-dominant fallback
  all fail to fire.
- The OCR / parser layers can return an empty payload or a
  review-bound payload that leads to `validation_status = "empty"` or
  `validation_status = "review_ocr_quality"`, in which case the
  classifier never receives strong cues.

Buckets B2 and B4 are the safe-default branch. Buckets B3, B5, B7 are
upstream OCR / parser layer signals. Bucket B6 is a coverage signal
that the Cyrillic-dominant fallback failed to absorb. Buckets B1 and B8
are workflow / UI signals.

## I. Is This Workflow Confusion, Classifier Limitation, OCR Routing
Limitation, Or Unsupported Document Type?

The answer is **bucket-dependent** and requires operator input from the
decision tree.

| Class | Mapped buckets |
| --- | --- |
| operator workflow / UI confusion | B1, B8 |
| classifier coverage limitation | B2, B4, B6 |
| OCR routing / quality limitation | B3, B5 |
| parser / extractor limitation | B7 |

## J. Is A Code Fix Justified?

**Not yet, and not in this block.** Justification requires the operator
to record the matched bucket id first. Buckets B1 and B8 never justify
a code fix. Buckets B2, B3, B4, B6, B7 may justify a follow-up
**evaluation-only** block, which would itself need to clear before any
behavior change is even considered. Bucket B5 has historic parks
already covering the layout-heavy story.

## K. Safety / Privacy Invariants

| Invariant | Value |
| --- | :-: |
| `block_mode` | `diagnosis_only_real_doc_unknown_triage` |
| `triggered_by_real_operator_signal` | **true** |
| `operator_supplied_real_document_artifact` | **false** |
| `real_document_inspected_by_assistant` | **false** |
| `raw_text_inspected` | **false** |
| `raw_ocr_text_inspected` | **false** |
| `raw_filename_inspected` | **false** |
| `private_paths_inspected` | **false** |
| `phi_inspected` | **false** |
| `real_pdf_committed` | **false** |
| `implementation_started` | **false** |
| `new_helper_created` | **false** |
| `direct_implementation_recommended` | **false** |
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
| `threshold_scoring_changed` | **false** |
| `parser_behavior_changed` | **false** |
| `fallback_behavior_changed` | **false** |
| `cue_pack_changed` | **false** |
| `cue_expansion_recommended` | **false** |
| `db_schema_changed` | **false** |
| `migration_created` | **false** |
| `migration_executed` | **false** |
| `persistence_code_changed` | **false** |
| `clinical_behavior_changed` | **false** |
| `ddi_behavior_changed` | **false** |
| `terminology_behavior_changed` | **false** |
| `private_adapter_implemented` | **false** |
| `concrete_adapters_implemented` | **false** |
| `runtime_wiring_added` | **false** |
| `external_api_used` | **false** |
| `private_data_accessed` | **false** |
| `source_documents_opened` | **false** |
| `raw_text_read` | **false** |
| `raw_text_printed` | **false** |
| `raw_ocr_text_read` | **false** |
| `raw_ocr_text_printed` | **false** |
| `raw_filenames_read` | **false** |
| `raw_filenames_printed` | **false** |
| `private_paths_printed` | **false** |
| `secrets_printed` | **false** |
| `licensed_rows_read` | **false** |
| `licensed_rows_exposed` | **false** |
| `private_license_ack_read` | **false** |
| `private_config_read` | **false** |
| `runtime_db_accessed` | **false** |
| `tags_touched` | **false** |
| `v1_release_preserved` | **true** |
| `local_only_default` | **true** |
| `review_bound_default` | **true** |
| `external_api_blocked_default` | **true** |
| `auto_accept_allowed_default` | **false** |

## L. Operator Action Required (Local)

The next concrete action is on the operator, not the assistant.

1. Open the failing run card in the local UI.
2. Expand the advanced diagnostic panel for that card.
3. Read the 20 advanced diagnostic fields off the card. Must not copy
   any raw text, raw filename, or private path into any public report.
4. Apply the decision tree in section E to map signals to a single
   bucket id (B1 through B8).
5. Record the matched bucket id in a private operator note (gitignored
   location only).
6. If the matched bucket has a conditional evaluation block in the
   section F table, the operator may explicitly request that
   evaluation-only block as the next MedAI work.
7. If the matched bucket is B1 or B8, no code change is justified and
   freeze-maintenance posture continues.

## M. Recommended Next Block

- **Primary:** `OPERATOR-RUN-DECISION-TREE-LOCALLY_THEN_BUCKET-CONDITIONAL-EVAL-OR-FREEZE-MAINTENANCE-ONLY`.
- **Conditional follow-up:** the bucket-conditional evaluation block id
  from the section F table, only after the operator records the
  matched bucket id.
- **Fallback:** `FREEZE-MAINTENANCE-ONLY` continues if the matched
  bucket is B1 or B8, or if the operator chooses to defer.

Forbidden in any follow-up block:

- starting V2 implementation directly;
- modifying classifier rules, OCR gate thresholds, parser support, or
  cue packs;
- reopening cue expansion (explicitly **NOT** recommended);
- reopening terminology / private adapter implementation;
- touching any existing parking / freeze tag;
- committing the real PDF, raw text, raw OCR text, raw filenames,
  private paths, PHI, or runtime DB rows.

V1 frozen release at `7ef8ffd` continues to be the durable shipped
artifact.
