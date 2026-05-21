# MEDAI-REAL-DOC-B4-PATHOLOGY-FAMILY-EVAL-01 — Short Summary

Evaluation-only follow-up to the real-document Unknown triage. The
operator visually identified the failing document as a pathology /
dermatopathology consultation report, matching bucket B4 (unsupported
or under-covered document family). This block inspects the existing
classifier and family registry, catalogs safe structural cues, and
recommends a future reports-only / synthetic-only spec block. No
classifier change. No cue expansion. No threshold change. No OCR
routing change. No real document inspected.

## Deliverables

- `scripts/run_medai_real_doc_b4_pathology_family_eval_01.py` —
  reports-only audit.
- `tests/test_medai_real_doc_b4_pathology_family_eval_01.py` — focused
  tests.
- 3 public-safe reports.

## Key findings

| Layer | Today's pathology coverage |
| --- | --- |
| Core `document_classifier.py` | **none** — emits `unknown_medical` (confidence 0.30) for pathology-shaped synthetic text |
| Metadata `document_type_registry.py` | **partial** — `PATHOLOGY_REPORT_LABEL` rule with 3 cue keys (specimen, microscopic description, pathology/final diagnosis), threshold 2, 4 language packs (english, russian, polish, albanian) |
| UI primary `Document type` card | shows `Unknown` (reads core layer) |
| UI secondary family badge | can show `Pathology report` (reads metadata layer) when threshold met |
| Auto-accept | **false** (review-bound) |

## Verdict

| Field | Value |
| --- | --- |
| `block_mode` | `evaluation_only_pathology_family_audit` |
| `evaluation_only` | true |
| `synthetic_only` | true |
| `matched_unknown_triage_bucket` | B4 |
| `core_classifier_has_pathology_cue` | false |
| `metadata_family_registry_has_pathology_rule` | true (partial) |
| `future_implementation_justified` | true (spec-only) |
| `recommended_controlled_family_name` | `pathology_report` |
| `recommended_subtype_field_name` | `pathology_subtype` |
| `cue_expansion_recommended` | false |
| `classifier_changed` | false |
| `behavior_changed_in_this_block` | false |
| `v1_release_preserved` | true |
| Next recommended block | **`MEDAI-PATHOLOGY-REPORT-FAMILY-SPEC-01_OR_FREEZE-MAINTENANCE-ONLY`** |

## Future spec recommendation, if approved

- **Block id:** `MEDAI-PATHOLOGY-REPORT-FAMILY-SPEC-01`.
- **Mode:** reports-only / synthetic-only.
- **Must not include:** real document text, real OCR text, real
  filename, real PDF, real screenshot, real diagnosis, cue expansion,
  threshold change, auto-accept enablement, classifier rule change,
  OCR routing change, external API use.
- **May include:** planning-level cue inventory, planning-level
  extraction field inventory, planning-level subtype controlled
  vocabulary, public-safe synthetic structural examples (generic,
  fake content only).

## Doctrine

Cue expansion remains explicitly **NOT** recommended. Private adapter
implementation remains blocked. V1 frozen release at `7ef8ffd`
continues to be the durable shipped artifact. Freeze-maintenance
posture remains intact.

## Progress

- Whole MedAI project: **~97.9%** done / ~2.1% remaining.
