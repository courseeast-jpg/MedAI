# Vertex Semantic Review Decision Audit

_Read-only audit. No live provider call. No active MKB records are created._

- Provider: route=`vertex`, model=`gemini-2.5-flash-lite`
- Total decisions: **15** | Accepted for review: **13** | Rejected: **1** | Deferred: **1**

## Package-family breakdown

- cytology_pathology_narrative: 3
- mixed_narrative_numeric_result: 4
- portal_result_cards: 4
- urinalysis_table_like_lab: 4

## Safety status

- active_written_count: **0**
- active_mkb_record_created_count: **0**
- auto_accept: **False**
- review_required: **True**
- hallucinated_field_count: **0**

## Decisions (evidence & provenance)

| Family | Status | Anchor | Source evidence | Route/Model | Source report | Audit reason |
| --- | --- | --- | --- | --- | --- | --- |
| cytology_pathology_narrative | review_draft | cyto_a1 | tests ordered section present | vertex/gemini-2.5-flash-lite | reports/medai_vertex_semantic_package_remaining_families_live_15p_d/live_remaining_families_results.json | operator accepted finding for review queue only; no active write |
| cytology_pathology_narrative | review_draft | cyto_a1 | diagnosis section present | vertex/gemini-2.5-flash-lite | reports/medai_vertex_semantic_package_remaining_families_live_15p_d/live_remaining_families_results.json | operator accepted finding for review queue only; no active write |
| cytology_pathology_narrative | review_draft | cyto_a1 | recommendation section present | vertex/gemini-2.5-flash-lite | reports/medai_vertex_semantic_package_remaining_families_live_15p_d/live_remaining_families_results.json | operator accepted finding for review queue only; no active write |
| urinalysis_table_like_lab | rejected | ua_a1 | table rows grouped under urinalysis table | vertex/gemini-2.5-flash-lite | reports/medai_vertex_semantic_package_remaining_families_live_15p_d/live_remaining_families_results.json | operator rejected finding; no active write |
| urinalysis_table_like_lab | review_draft | ua_a1 | table rows grouped under urinalysis table | vertex/gemini-2.5-flash-lite | reports/medai_vertex_semantic_package_remaining_families_live_15p_d/live_remaining_families_results.json | operator accepted finding for review queue only; no active write |
| urinalysis_table_like_lab | review_draft | ua_a1 | table rows grouped under urinalysis table | vertex/gemini-2.5-flash-lite | reports/medai_vertex_semantic_package_remaining_families_live_15p_d/live_remaining_families_results.json | operator accepted finding for review queue only; no active write |
| urinalysis_table_like_lab | review_draft | ua_a1 | table rows grouped under urinalysis table | vertex/gemini-2.5-flash-lite | reports/medai_vertex_semantic_package_remaining_families_live_15p_d/live_remaining_families_results.json | operator accepted finding for review queue only; no active write |
| portal_result_cards | deferred | portal_a1 | card labels grouped as portal result cards | vertex/gemini-2.5-flash-lite | reports/medai_vertex_semantic_package_comparison_live_15p_c/live_comparison_result.json | operator deferred finding; no active write |
| portal_result_cards | review_draft | portal_a1 | card labels grouped as portal result cards | vertex/gemini-2.5-flash-lite | reports/medai_vertex_semantic_package_comparison_live_15p_c/live_comparison_result.json | operator accepted finding for review queue only; no active write |
| portal_result_cards | review_draft | portal_a1 | card labels grouped as portal result cards | vertex/gemini-2.5-flash-lite | reports/medai_vertex_semantic_package_comparison_live_15p_c/live_comparison_result.json | operator accepted finding for review queue only; no active write |
| portal_result_cards | review_draft | portal_a1 | card labels grouped as portal result cards | vertex/gemini-2.5-flash-lite | reports/medai_vertex_semantic_package_comparison_live_15p_c/live_comparison_result.json | operator accepted finding for review queue only; no active write |
| mixed_narrative_numeric_result | review_draft | mixed_a1 | narrative section present | vertex/gemini-2.5-flash-lite | reports/medai_vertex_semantic_package_remaining_families_live_15p_d/live_remaining_families_results.json | operator accepted finding for review queue only; no active write |
| mixed_narrative_numeric_result | review_draft | mixed_a1 | numeric/result section present | vertex/gemini-2.5-flash-lite | reports/medai_vertex_semantic_package_remaining_families_live_15p_d/live_remaining_families_results.json | operator accepted finding for review queue only; no active write |
| mixed_narrative_numeric_result | review_draft | mixed_a1 | numeric/result section present | vertex/gemini-2.5-flash-lite | reports/medai_vertex_semantic_package_remaining_families_live_15p_d/live_remaining_families_results.json | operator accepted finding for review queue only; no active write |
| mixed_narrative_numeric_result | review_draft | mixed_a1 | numeric/result section present | vertex/gemini-2.5-flash-lite | reports/medai_vertex_semantic_package_remaining_families_live_15p_d/live_remaining_families_results.json | operator accepted finding for review queue only; no active write |

## Exports (read-only)

- JSON: `reports/medai_vertex_semantic_review_decision_audit_export_15s/decision_audit_export.json`
- CSV: `reports/medai_vertex_semantic_review_decision_audit_export_15s/decision_audit_export.csv`
- MARKDOWN: `reports/medai_vertex_semantic_review_decision_audit_export_15s/decision_audit_summary.md`

_Exports are read-only. No active MKB records are created._
