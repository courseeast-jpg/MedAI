# MEDAI-PATHOLOGY-REPORT-FAMILY-SPEC-01 — Reports-Only / Synthetic-Only Pathology Report Family Spec

Reports-only / synthetic-only family specification for pathology /
dermatopathology / biopsy pathology reports. Defines the controlled
family name, the subtype controlled vocabulary, the safe structural
cue groups (planning only), the future extraction field contract, the
review-bound default behavior, the Dermatology MKB placement doctrine,
the AI-agent interpretation boundary, the photomicrograph boundary,
and the future implementation gates. **No** classifier change. **No**
cue expansion. **No** threshold change. **No** OCR routing change.
**No** parser change. **No** UI / Streamlit / launcher / startup /
config change. **No** persistence change. **No** terminology / private
adapter work. **No** external API call. **No** real PDF / screenshot /
diagnosis / raw text / raw OCR text / raw filename / private path /
PHI / runtime DB row read or printed. V1 frozen release at `7ef8ffd`
remains preserved. Cue expansion remains explicitly **NOT**
recommended.

## A. Scope And Non-Scope

**In scope**

- Define the controlled future family name and the controlled subtype
  vocabulary.
- Catalog the safe structural cue groups at planning level only.
- Define the future review-bound extraction field contract.
- Define the Dermatology MKB placement doctrine.
- Define the AI-agent interpretation boundary.
- Define the photomicrograph boundary.
- Catalog the future implementation gates.
- Provide synthetic-only structural examples if needed (no real
  document text).
- Provide focused tests verifying the spec content.

**Out of scope**

- Adding pathology cues to `document_classification/document_classifier.py`.
- Editing cue groups in `app/document_type_registry.py`.
- Changing thresholds.
- Changing OCR routing.
- Changing the parser.
- Changing the UI, launchers, startup, or config.
- Changing persistence, DB schema, or migrations.
- Implementing AI-agent interpretation.
- Implementing photomicrograph interpretation.
- Reading the operator's real PDF, screenshot, raw text, raw OCR text,
  raw filename, private paths, PHI, or runtime DB rows.
- Reading licensed terminology rows or private config contents.
- Calling an external API.
- Creating, moving, or deleting tags.

## B. Prior B4 Evaluation Summary

The preceding block `MEDAI-REAL-DOC-B4-PATHOLOGY-FAMILY-EVAL-01`
(commit `12fa16a`) established:

- the core `document_classifier.py` has **no** pathology cues and
  emits `unknown_medical` on pathology-shaped synthetic text;
- the metadata `document_type_registry.py` already declares a
  `PATHOLOGY_REPORT_LABEL` rule (threshold 2) covering 4 language
  packs (english, russian, polish, albanian) with 3 cue keys
  (`specimen_section`, `microscopic_description_section`,
  `pathology_conclusion_section`);
- a future implementation is justified as a reports-only /
  synthetic-only spec block (this block), not as a behavior change.

V1 frozen release anchor: `7ef8ffd`. Freeze-maintenance-only anchor:
`fb730f2`. Real-doc unknown triage anchor: `324682c`. B4 evaluation
anchor: `12fa16a`.

## C. Controlled Family Naming

| Choice | Decision |
| --- | --- |
| Controlled future family name | **`pathology_report`** |
| Separate top-level `dermatopathology_report` family | rejected |
| Separate top-level `biopsy_pathology_report` family | rejected |
| Separate top-level `histopathology_report` family | rejected |
| Subtype field name | **`pathology_subtype`** |

Rationale: matches the existing `PATHOLOGY_REPORT_LABEL` constant and
the existing metadata family rule. Avoids fragmenting the family
registry across subspecialities. Subspeciality membership is
represented through the `pathology_subtype` controlled vocabulary
below.

## D. Pathology Subtype Vocabulary

The controlled subtype values (planning level only, not implemented):

| # | Value | Notes |
| -: | --- | --- |
| 1 | `dermatopathology` | skin biopsy / dermatology consultation |
| 2 | `biopsy` | general biopsy outside dermatology |
| 3 | `cytology` | cytology preparations and reports |
| 4 | `frozen_section` | intraoperative frozen-section reports |
| 5 | `consultation_pathology` | external consultation reports |
| 6 | `unspecified_pathology` | conservative fallback when subtype cannot be inferred from safe cues |

`unspecified_pathology` is the conservative default. The subtype is
review-bound; auto-accept is **not allowed**.

## E. Safe Structural Cue Groups (Planning Only)

Planning-level cue groups for a future spec. **No cue is added to
classifier code in this block.**

**Core cue candidates (already partially covered by the existing
metadata family rule):**

- `specimen_section`
- `microscopic_description_section`
- `pathology_conclusion_section`

**Additional cue candidates (operator-observed in the real document
structure, mapped only conceptually here):**

- `gross_description`
- `clinical_diagnosis`
- `signs_and_symptoms`
- `reason_for_biopsy`
- `photomicrograph_section`
- `margin_or_edge_or_inked_edge_comment`
- `icd_code`
- `pathology_consultation_header`

| Constraint | Value |
| --- | :-: |
| Cue keys added to runtime classifier code in this block | **none** |
| Classifier thresholds altered in this block | **no** |
| Auto-accept behavior altered in this block | **no** |
| Current Unknown behavior altered in this block | **no** |

## F. Future Review-Bound Extraction Field Contract

Eighteen review-bound fields defined at planning level only. Every
field has `review_required=true`, `auto_accept_allowed=false`,
`source_facts_only=true`, and
`ai_interpretation_allowed_in_classifier_layer=false`.

| # | Field | Type |
| -: | --- | --- |
|  1 | `document_family` | controlled vocabulary |
|  2 | `pathology_subtype` | controlled vocabulary |
|  3 | `specialty_domain` | controlled vocabulary |
|  4 | `specimen` | structural text extract |
|  5 | `anatomical_site` | structural text extract |
|  6 | `clinical_diagnosis_or_indication` | structural text extract |
|  7 | `signs_and_symptoms` | structural text extract |
|  8 | `gross_description` | structural text extract |
|  9 | `microscopic_description` | structural text extract |
| 10 | `final_diagnosis` | structural text extract |
| 11 | `diagnosis_code_family` | controlled vocabulary |
| 12 | `diagnosis_code_value` | controlled identifier |
| 13 | `margin_or_edge_comment` | structural text extract |
| 14 | `photomicrograph_present` | boolean presence flag |
| 15 | `source_document_reference` | anonymous identifier |
| 16 | `extraction_confidence` | float (0..1) |
| 17 | `review_required` | boolean default `true` |
| 18 | `auto_accept_allowed` | boolean default `false` |

Required defaults across every future implementation:

| Default | Value |
| --- | :-: |
| `review_required` | **true** |
| `auto_accept_allowed` | **false** |
| `source_facts_only` | **true** |
| `ai_interpretation_allowed` (classifier / extraction layer) | **false** |
| `ai_interpretation_tier` | hypothesis or comment only, in a future separate agent block |

## G. Dermatology MKB Placement Doctrine

Pathology reports may map to a specialty / domain such as Dermatology
only when safe source cues support the mapping. The mapping is
**advisory and review-bound**.

| Rule | Doctrine |
| --- | --- |
| 1 | dermatopathology subtype may map to Dermatology section |
| 2 | mapping must be review-bound |
| 3 | no automatic clinical inference |
| 4 | no treatment recommendation |
| 5 | no prognosis inference |
| 6 | no autonomous medical conclusion |

The mapping cannot drive any automatic clinical decision in the
classifier / extraction layer. Any clinical implication is operator
work.

## H. AI-Agent Interpretation Boundary

Future workflow defined conceptually only. **Not implemented in this
block.**

| Rule | Doctrine |
| --- | --- |
| 1 | save source facts as a source-derived record |
| 2 | optionally send de-identified extracted content to an AI agent only in a separate future block |
| 3 | external AI default blocked |
| 4 | operator confirmation required |
| 5 | PII / privacy scrub required |
| 6 | AI output must be stored as hypothesis or comment, not active source fact |
| 7 | no AI interpretation implementation in this block |

`ai_interpretation_implemented`: **false**.

## I. Photomicrograph Boundary

Future options defined conceptually only. **Not implemented in this
block.**

| Option | Doctrine |
| --- | --- |
| 1 | record image presence only |
| 2 | store image reference or attachment only if privacy-approved |
| 3 | no image interpretation in this family classifier block |
| 4 | image interpretation requires a separate visual-AI safety spec |

`photomicrograph_interpretation_implemented`: **false**.

## J. Safety / Privacy Invariants

| Invariant | Value |
| --- | :-: |
| `block_mode` | `reports_only_synthetic_only_family_spec` |
| `reports_only` | **true** |
| `synthetic_only` | **true** |
| `spec_only` | **true** |
| `pathology_family_spec_created` | **true** |
| `family_name` | `pathology_report` |
| `subtype_field` | `pathology_subtype` |
| `pathology_family_support_justified` | **true** |
| `implementation_started` | **false** |
| `new_helper_created` | **false** |
| `direct_implementation_recommended` | **false** |
| `behavior_changed_in_this_block` | **false** |
| `classifier_changed` | **false** |
| `classifier_cue_added` | **false** |
| `classifier_rule_changed` | **false** |
| `cue_expansion_performed` | **false** |
| `cue_expansion_recommended` | **false** |
| `threshold_changed` | **false** |
| `threshold_scoring_changed` | **false** |
| `ocr_routing_changed` | **false** |
| `parser_changed` | **false** |
| `parser_behavior_changed` | **false** |
| `runtime_behavior_changed` | **false** |
| `app_main_changed` | **false** |
| `streamlit_code_changed` | **false** |
| `ui_changed` | **false** |
| `launcher_changed` / `installer_changed` / `deployment_script_changed` | **false** |
| `startup_config_changed` | **false** |
| `extraction_changed` | **false** |
| `ocr_changed` | **false** |
| `fallback_behavior_changed` | **false** |
| `cue_pack_changed` | **false** |
| `db_schema_changed` | **false** |
| `migration_created` / `migration_executed` | **false** |
| `persistence_code_changed` | **false** |
| `clinical_behavior_changed` / `ddi_behavior_changed` / `terminology_behavior_changed` | **false** |
| `private_adapter_implemented` / `concrete_adapters_implemented` / `runtime_wiring_added` | **false** |
| `external_api_used` | **false** |
| `private_data_accessed` | **false** |
| `source_documents_opened` | **false** |
| `real_pdf_committed` | **false** |
| `real_screenshot_committed` | **false** |
| `real_diagnosis_printed` | **false** |
| `raw_text_read` / `raw_text_printed` | **false** |
| `raw_ocr_text_read` / `raw_ocr_text_printed` | **false** |
| `raw_filenames_read` / `raw_filenames_printed` | **false** |
| `private_paths_printed` | **false** |
| `secrets_printed` | **false** |
| `phi_printed` / `phi_inspected` | **false** |
| `licensed_rows_read` / `licensed_rows_exposed` | **false** |
| `license_ack_read` / `private_license_ack_read` | **false** |
| `private_config_read` | **false** |
| `runtime_db_accessed` | **false** |
| `ai_interpretation_implemented` | **false** |
| `photomicrograph_interpretation_implemented` | **false** |
| `tags_touched` / `tags_created` / `tags_modified` | **false** |
| `review_required_default` | **true** |
| `auto_accept_allowed_default` | **false** |
| `source_facts_only` | **true** |
| `v1_release_preserved` | **true** |
| `local_only_default` | **true** |
| `review_bound_default` | **true** |
| `external_api_blocked_default` | **true** |

## K. Future Implementation Gates

Eleven gates govern any future block that builds on this spec. Every
gate must clear before behavior changes.

| Gate | Rule |
| :-: | --- |
| A | A synthetic-coverage block must clear before any real classifier change |
| B | Cue expansion remains blocked by the V2 default-off status registry (entry `cue_expansion`, status `blocked`) |
| C | Threshold change requires a separate approved block |
| D | Auto-accept must remain `false` through the entire extraction field contract |
| E | Clinical interpretation layer requires a separate clinical safety spec |
| F | Dermatology MKB mapping must remain review-bound |
| G | AI-agent interpretation requires a separate privacy-approved spec |
| H | Photomicrograph interpretation requires a separate visual-AI safety spec |
| I | Any real document is blocked from commit until the privacy scanner clears an explicit synthetic-only fixture substitution |
| J | External API remains blocked by default |
| K | Runtime DB remains row-blind in public reports |

## L. Validation Matrix

V2 validation matrix categories referenced (all 8 from
`V2-VALIDATION-HARNESS-01`):

| Category | Conformance |
| --- | --- |
| A — Contract import + side-effect safety | no module side effects added |
| B — Contract inventory conformance | no contract change |
| C — Safety profile conformance | default safety profile unchanged |
| D — Terminology aggregate-only conformance | terminology aggregate-only doctrine preserved |
| E — Review / HITL conformance | review-bound default preserved; auto-accept blocked |
| F — Reports privacy conformance | 3 reports pass privacy check |
| G — Runtime non-modification conformance | no runtime / `app/main.py` / launcher / preflight / config / Streamlit / classifier / family-registry / OCR-gate change |
| H — Parking / freeze preservation conformance | freeze-maintenance posture intact; all existing parking / freeze tags unchanged |

## M. Recommended Next Block

- **Primary:** `MEDAI-PATHOLOGY-REPORT-FAMILY-SYNTHETIC-COVERAGE-01` —
  reports-only / synthetic-only coverage audit that exercises the
  metadata family rule against a small set of synthetic
  pathology-shaped public-safe fixtures and reports aggregate match
  rates per cue key. Still no classifier change, still no cue
  expansion, still no auto-accept change.
- **Fallback:** `FREEZE-MAINTENANCE-ONLY` if the operator defers the
  synthetic coverage block.
- **Alternative:** `V2-ROADMAP-04` only if a new strategic signal
  appears.

Forbidden in any follow-up block:

- adding pathology cues to `document_classifier.py`;
- editing cue groups in `document_type_registry.py`;
- changing thresholds, OCR routing, parsers, or extractors;
- enabling auto-accept on any pathology extraction field;
- implementing AI-agent interpretation;
- implementing photomicrograph interpretation;
- reopening cue expansion (explicitly **NOT** recommended);
- reopening terminology / private adapter implementation;
- touching any existing parking / freeze tag;
- committing the operator's real PDF, real screenshot, real diagnosis,
  raw text, raw OCR text, raw filenames, private paths, PHI, or
  runtime DB rows.

V1 frozen release at `7ef8ffd` continues to be the durable shipped
artifact. Freeze-maintenance posture remains intact.
