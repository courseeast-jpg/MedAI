# 15X Vertex calibration batch matrix

- Cost table: Conservative local estimate: input $0.0001/1K tokens, output $0.0004/1K tokens for gemini-2.5-flash-lite. Not from a billing API; billing_check_pending=true.
- Category breakdown: `{'portal_result_cards': 2, 'cytology_pathology_narrative': 2, 'urinalysis_table_like_lab': 2, 'mixed_narrative_numeric_result': 1, 'short_clinical_note_negation': 1, 'short_clinical_note_uncertainty': 1, 'medication_mention_no_ddi': 1, 'bilingual_cyrillic_snippet': 1, 'sparse_low_information_result': 1, 'multi_section_explicit_unknowns': 1, 'abnormal_numeric_with_units': 1, 'normal_numeric_with_units': 1}`

| Metric | Value |
| --- | --- |
| status | `PASS` |
| fixture_count | `15` |
| live_call_count | `15` |
| provider_response_received_count | `15` |
| schema_validation_pass_count | `15` |
| verbatim_evidence_anchor_pass_count | `15` |
| source_visible_body_preserved_count | `15` |
| evidence_anchor_preserved_count | `15` |
| candidate_facts_separated_count | `15` |
| unknown_values_explicit_count | `15` |
| uncertainty_flags_visible_count | `15` |
| posted_body_allowed_top_level_keys_only_count | `15` |
| review_required_count | `15` |
| hallucinated_field_count | `0` |
| total_prompt_tokens | `4950` |
| total_output_tokens | `2384` |
| total_token_count_all_calls | `7334` |
| estimated_total_cost_usd_all_calls | `0.0014486` |
| estimated_cost_ceiling_usd | `0.00512` |
| failed_call_count | `0` |
| auto_accept_true_count | `0` |
| active_written_count | `0` |
| active_mkb_record_created_count | `0` |
| live_call_made | `True` |
| external_api_used | `True` |
| privacy_result | `passed` |
| billing_check_pending | `True` |
| stopped_early | `False` |
| stop_reason | `` |

Synthetic/redacted fixtures only; bounded <=20 live calls; no active writes; no auto-accept.
