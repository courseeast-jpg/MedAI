# MEDAI-REAL-DOC-UNKNOWN-TRIAGE-01 — Short Summary

Diagnosis-only triage block triggered by a real local operator signal:
a real document returned Unknown / rejected / no useful result through
the frozen MedAI pipeline. This block produces a public-safe diagnostic
framework. No runtime behavior change. No classifier change. No OCR
routing change. No threshold or cue change. No persistence change. No
external API call.

## Deliverables

- `scripts/run_medai_real_doc_unknown_triage_01.py` — reports-only
  audit.
- `tests/test_medai_real_doc_unknown_triage_01.py` — focused tests.
- 3 public-safe reports.

## Failure bucket taxonomy (8)

| ID | Bucket |
| :-: | --- |
| B1 | `file_not_queued_or_run_not_executed` |
| B2 | `text_extracted_but_classifier_cues_insufficient` |
| B3 | `ocr_fallback_needed_but_not_triggered` |
| B4 | `document_family_unsupported` |
| B5 | `image_or_table_or_layout_heavy_document` |
| B6 | `language_or_script_mismatch` |
| B7 | `parser_unsupported` |
| B8 | `ui_run_state_confusion` |

## Verdict

| Field | Value |
| --- | --- |
| `block_mode` | `diagnosis_only_real_doc_unknown_triage` |
| `triggered_by_real_operator_signal` | true |
| `operator_supplied_real_document_artifact` | false |
| `real_document_inspected_by_assistant` | false |
| `raw_text_inspected` | false |
| `raw_filename_inspected` | false |
| `phi_inspected` | false |
| `triage_framework_created` | true |
| `failure_bucket_taxonomy_created` | true |
| `decision_tree_created` | true |
| `bucket_conditional_next_block_table_created` | true |
| `operator_action_required` | true |
| `code_fix_justified_yet` | false |
| `runtime_behavior_changed` | false |
| `classifier_changed` | false |
| `ocr_changed` / `ocr_routing_changed` | false |
| `cue_pack_changed` / `cue_expansion_recommended` | false |
| `v1_release_preserved` | true |
| Next recommended block | **`OPERATOR-RUN-DECISION-TREE-LOCALLY_THEN_BUCKET-CONDITIONAL-EVAL-OR-FREEZE-MAINTENANCE-ONLY`** |

## Operator next action

1. Open the failing run card locally.
2. Expand the advanced diagnostic panel (20 public-safe fields).
3. Apply the 5-step decision tree in section E of the long report.
4. Record the matched bucket id (B1..B8) in a private operator note.
5. If the bucket maps to a conditional evaluation block, request that
   evaluation-only follow-up. If the bucket is B1 or B8, no code change
   is justified.

## Doctrine

Cue expansion remains explicitly **NOT** recommended. Private adapter
implementation remains blocked. V1 frozen release at `7ef8ffd`
continues to be the durable shipped artifact. Freeze-maintenance posture
remains intact.

## Progress

- Whole MedAI project: **~97.8%** done / ~2.2% remaining.
