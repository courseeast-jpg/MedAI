# MEDAI-PATHOLOGY-REPORT-FAMILY-SPEC-01 — Short Summary

Reports-only / synthetic-only family spec for pathology /
dermatopathology / biopsy pathology reports. Follow-up to
`MEDAI-REAL-DOC-B4-PATHOLOGY-FAMILY-EVAL-01`. Defines the controlled
family name, subtype vocabulary, safe cue groups (planning only),
review-bound extraction field contract, Dermatology MKB placement
doctrine, AI-agent interpretation boundary, photomicrograph boundary,
and future implementation gates. No classifier change. No cue
expansion. No threshold change. No OCR routing change. No parser
change. No UI change. No persistence change. No AI interpretation. No
photomicrograph interpretation. No real document committed.

## Deliverables

- `scripts/run_medai_pathology_report_family_spec_01.py` —
  reports-only audit.
- `tests/test_medai_pathology_report_family_spec_01.py` — focused
  tests.
- 3 public-safe reports.

## Spec at a glance

| Field | Value |
| --- | --- |
| `family_name` | `pathology_report` |
| `subtype_field` | `pathology_subtype` |
| Subtype vocabulary (6) | `dermatopathology`, `biopsy`, `cytology`, `frozen_section`, `consultation_pathology`, `unspecified_pathology` |
| Core cue candidates (3) | `specimen_section`, `microscopic_description_section`, `pathology_conclusion_section` |
| Additional cue candidates (8) | `gross_description`, `clinical_diagnosis`, `signs_and_symptoms`, `reason_for_biopsy`, `photomicrograph_section`, `margin_or_edge_or_inked_edge_comment`, `icd_code`, `pathology_consultation_header` |
| Future extraction fields (18) | see Section F of the long report |
| Future implementation gates (11) | see Section K of the long report |

## Verdict

| Field | Value |
| --- | --- |
| `block_mode` | `reports_only_synthetic_only_family_spec` |
| `pathology_family_spec_created` | true |
| `pathology_family_support_justified` | true |
| `cue_expansion_performed` | false |
| `cue_expansion_recommended` | false |
| `classifier_changed` | false |
| `threshold_changed` | false |
| `ocr_routing_changed` | false |
| `parser_changed` | false |
| `runtime_behavior_changed` | false |
| `ui_changed` | false |
| `db_schema_changed` | false |
| `external_api_used` | false |
| `private_data_accessed` | false |
| `real_pdf_committed` | false |
| `real_screenshot_committed` | false |
| `phi_printed` | false |
| `ai_interpretation_implemented` | false |
| `photomicrograph_interpretation_implemented` | false |
| `review_required_default` | true |
| `auto_accept_allowed_default` | false |
| `source_facts_only` | true |
| `v1_release_preserved` | true |
| Next recommended block | **`MEDAI-PATHOLOGY-REPORT-FAMILY-SYNTHETIC-COVERAGE-01_OR_FREEZE-MAINTENANCE-ONLY`** |

## Doctrine

Cue expansion remains explicitly **NOT** recommended. Private adapter
implementation remains blocked. V1 frozen release at `7ef8ffd`
continues to be the durable shipped artifact. Freeze-maintenance
posture remains intact.

## Progress

- Whole MedAI project: **~98.0%** done / ~2.0% remaining.
