# MEDAI Final Post Document-Type Family Validation Snapshot

Conclusion: `medai_final_post_doc_type_family_validation_snapshot_ready`

Branch: `clinical-knowledge-architecture`

HEAD commit: `496bb4e`

Commit provenance is public-report safe and uses short hashes.

## Included Blocks

| Block | Commit | Summary |
| --- | --- | --- |
| MEDAI-DOC-TYPE-FAMILY-03 | `d5e07f4` | Added conservative Latin lab-structure cues. |
| MEDAI-DOC-TYPE-FAMILY-04 | `496bb4e` | Validated FAMILY-03 on a larger 507-file anonymized local slice. |

## 80-File Bang-Folder Result

| Family | Before | After |
| --- | ---: | ---: |
| Lab result | 15 | 78 |
| Urinalysis | 1 | 1 |
| Unknown | 64 | 1 |

- all_80_review_bound: `true`
- accepted_count: `0`
- auto_accept_allowed_count: `0`
- external_api_used_count: `0`

## 507-File Larger Slice Result

| Family | Count |
| --- | ---: |
| Lab result | 330 |
| Urinalysis | 35 |
| Imaging report | 14 |
| Treatment plan | 4 |
| Medication plan | 11 |
| Clinical note | 2 |
| Administrative / Insurance | 4 |
| Discharge summary | 0 |
| Referral / Order | 0 |
| Procedure report | 0 |
| Pathology report | 0 |
| Unknown | 107 |

- total_files_evaluated: `507`
- all_507_review_bound: `true`
- accepted_count: `0`
- auto_accept_allowed_count: `0`
- external_api_used_count: `0`

## Unknown Buckets

| Bucket | Count |
| --- | ---: |
| insufficient_text_visibility | 75 |
| fallback_ran_but_no_family_match | 17 |
| ambiguous_below_threshold | 15 |

## False-Positive Risk Audit

- lab_vs_treatment_or_medication_ambiguous_candidates: `3`, all review-bound
- lab_vs_imaging_ambiguous_candidates: `3`, all review-bound
- lab_vs_admin_or_insurance_cue_overlap: `0`
- unknown_accepted_anomaly: `false`
- invalid_status_mapping_normalization: `false`
- family_03_generalizes_safely: `true`

## Validation Summary

- final CKA MVP validation: `passed`
- B07 validation: `passed`
- ROUTE-FIX validation: `passed`
- UI ops validation: `passed`
- UI boot validation: `passed`
- public report privacy checks: `passed`
- staged safety check: `passed`
- focused FAMILY/eval tests from FAMILY-04: `62 passed, 1 warning`
- UI/Russian/OCR/upload regression group from FAMILY-04: `157 passed`
- full pytest: `not run for PARK-19; skipped as not practical for a snapshot-only block after the larger 507-file validation and prior long full-suite timeout behavior`

## Safety And Privacy

- runtime_behavior_changed_in_park_block: `false`
- ocr_routing_changed: `false`
- ocr_engine_changed: `false`
- classifier_behavior_changed: `false`
- confidence_thresholds_changed: `false`
- confidence_scoring_changed: `false`
- auto_acceptance_changed: `false`
- clinical_interpretation_added: `false`
- lab_value_parsing_added: `false`
- medication_parsing_added: `false`
- dose_parsing_added: `false`
- ddi_logic_changed: `false`
- b07_terminology_changed: `false`
- route_fix_changed: `false`
- db_schema_changed: `false`
- command_allowlist_changed: `false`
- external_api_behavior_changed: `false`
- external_api_used: `false`
- raw_ocr_text_in_public_reports: `false`
- raw_document_text_in_public_reports: `false`
- raw_filenames_private_paths_in_public_reports: `false`
- source_documents_staged: `false`
- private_corpus_files_staged: `false`
- secrets_in_public_reports: `false`

## Recommendation

Leave remaining Unknown files review-bound. Run the next diagnostic only if operational coverage requires it, prioritizing text visibility and fallback-ran-no-family-match buckets before any further cue expansion.
