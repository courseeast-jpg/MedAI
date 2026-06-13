# Vertex Semantic Review Decisions — Operator Audit Summary (15S, read-only)

Read-only audit over the isolated 15R review-decision store. No active MKB writes; no auto-accept; no provider call.

- Decision records loaded: **15**
- Accepted for review: **13** | Rejected: **1** | Deferred: **1**
- Provider provenance: route=`vertex`, model=`gemini-2.5-flash-lite` (visible on 15/15)
- Hallucinated field count: **0** | Active MKB records: **0** | Active written count: **0** | Auto-accept: **0** true

## Per-family decision counts

- cytology_pathology_narrative: 3
- mixed_narrative_numeric_result: 4
- portal_result_cards: 4
- urinalysis_table_like_lab: 4

## Per-action decision counts

- accept_for_review: 13
- defer: 1
- reject: 1

## Decisions

| Decision | Family | Action | Status | Anchor | Source report | Audit reason |
| --- | --- | --- | --- | --- | --- | --- |
| dec_7313a2555be02155 | cytology_pathology_narrative | accept_for_review | review_draft | cyto_a1 | reports/medai_vertex_semantic_package_remaining_families_live_15p_d/live_remaining_families_results.json | operator accepted finding for review queue only; no active write |
| dec_69780ce21a387e20 | cytology_pathology_narrative | accept_for_review | review_draft | cyto_a1 | reports/medai_vertex_semantic_package_remaining_families_live_15p_d/live_remaining_families_results.json | operator accepted finding for review queue only; no active write |
| dec_8c7ca126aa8c4383 | cytology_pathology_narrative | accept_for_review | review_draft | cyto_a1 | reports/medai_vertex_semantic_package_remaining_families_live_15p_d/live_remaining_families_results.json | operator accepted finding for review queue only; no active write |
| dec_d8b82ebcb01be56f | urinalysis_table_like_lab | reject | rejected | ua_a1 | reports/medai_vertex_semantic_package_remaining_families_live_15p_d/live_remaining_families_results.json | operator rejected finding; no active write |
| dec_766cd2d0c58d53c6 | urinalysis_table_like_lab | accept_for_review | review_draft | ua_a1 | reports/medai_vertex_semantic_package_remaining_families_live_15p_d/live_remaining_families_results.json | operator accepted finding for review queue only; no active write |
| dec_57370b11183b5b7e | urinalysis_table_like_lab | accept_for_review | review_draft | ua_a1 | reports/medai_vertex_semantic_package_remaining_families_live_15p_d/live_remaining_families_results.json | operator accepted finding for review queue only; no active write |
| dec_538e0bc445144d5a | urinalysis_table_like_lab | accept_for_review | review_draft | ua_a1 | reports/medai_vertex_semantic_package_remaining_families_live_15p_d/live_remaining_families_results.json | operator accepted finding for review queue only; no active write |
| dec_ab51c84b93640e98 | portal_result_cards | defer | deferred | portal_a1 | reports/medai_vertex_semantic_package_comparison_live_15p_c/live_comparison_result.json | operator deferred finding; no active write |
| dec_7871c6b36ba7adea | portal_result_cards | accept_for_review | review_draft | portal_a1 | reports/medai_vertex_semantic_package_comparison_live_15p_c/live_comparison_result.json | operator accepted finding for review queue only; no active write |
| dec_c0594602ac9f679c | portal_result_cards | accept_for_review | review_draft | portal_a1 | reports/medai_vertex_semantic_package_comparison_live_15p_c/live_comparison_result.json | operator accepted finding for review queue only; no active write |
| dec_7174ce44bd6212ff | portal_result_cards | accept_for_review | review_draft | portal_a1 | reports/medai_vertex_semantic_package_comparison_live_15p_c/live_comparison_result.json | operator accepted finding for review queue only; no active write |
| dec_c53e57ee4debddbc | mixed_narrative_numeric_result | accept_for_review | review_draft | mixed_a1 | reports/medai_vertex_semantic_package_remaining_families_live_15p_d/live_remaining_families_results.json | operator accepted finding for review queue only; no active write |
| dec_e3e8c67ae03ef939 | mixed_narrative_numeric_result | accept_for_review | review_draft | mixed_a1 | reports/medai_vertex_semantic_package_remaining_families_live_15p_d/live_remaining_families_results.json | operator accepted finding for review queue only; no active write |
| dec_6b3d89d5148014e6 | mixed_narrative_numeric_result | accept_for_review | review_draft | mixed_a1 | reports/medai_vertex_semantic_package_remaining_families_live_15p_d/live_remaining_families_results.json | operator accepted finding for review queue only; no active write |
| dec_da10bb4eaa8b9da6 | mixed_narrative_numeric_result | accept_for_review | review_draft | mixed_a1 | reports/medai_vertex_semantic_package_remaining_families_live_15p_d/live_remaining_families_results.json | operator accepted finding for review queue only; no active write |
