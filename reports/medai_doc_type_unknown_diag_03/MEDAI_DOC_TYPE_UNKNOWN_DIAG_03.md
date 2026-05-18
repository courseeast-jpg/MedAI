# MEDAI-DOC-TYPE-UNKNOWN-DIAG-03 - Text Visibility Root-Cause Classifier

- branch: `clinical-knowledge-architecture`
- HEAD commit (short): `b8b46dbf030a`
- PARK-19 baseline commit (short): `ac466e0f9ab8`
- public_report_commit_hash_policy: `short_hashes_only`
- upstream diagnostics: `reports/medai_doc_type_unknown_diag_01/(public diagnostic)`, `reports/medai_doc_type_unknown_diag_02/(public diagnostic)`
- source report: `reports/medai_doc_type_family_04_larger_slice_validation/(public anonymized batch-eval per-file table)`
- total priority records analyzed: `52`
- generated_at: `2026-05-18T18:14:01.208517+00:00`

## Target subset counts

- language_script_visible_detector_unresolved: `31`
- likely_text_layer_issue: `21`

## Root-cause candidate counts

### language_script_visible_detector_unresolved

- script_visible_language_detector_gap: `0`
- latin_visible_language_unknown: `15`
- cyrillic_visible_language_unknown: `0`
- mixed_script_detector_gap: `0`
- numeric_or_table_heavy_language_detector_gap: `16`
- insufficient_safe_metadata: `0`
- leave_manual_review: `0`

### likely_text_layer_issue

- text_layer_too_short: `11`
- text_layer_present_but_low_signal: `0`
- table_structure_visible_but_text_insufficient: `10`
- image_like_with_partial_text: `0`
- no_safe_text_visibility_metadata: `0`
- leave_manual_review: `0`

## Next-action bucket counts

- candidate_language_detector_diagnostic: `31`
- candidate_text_layer_extraction_diagnostic: `21`
- candidate_ocr_routing_diagnostic: `0`
- leave_manual_review: `0`
- insufficient_metadata_for_next_action: `0`

## Implementation-block recommendation

- implementation_block_justified: `True`
- choice: `A` (A=language-detector, B=text-layer, C=OCR-routing, D=leave manual review)

Both diagnostics meet the threshold. 31 records point to candidate_language_detector_diagnostic and 21 to candidate_text_layer_extraction_diagnostic. Recommend A first (larger pool, upstream cause), then B.

## Deferred subsets (out of scope)

- fallback_ran_but_no_family_match: deferred to a future cue-coverage audit block per UNKNOWN-DIAG-02
- ambiguous_below_threshold: remains excluded from optimization; review-bound, no cue expansion

## Safety / Privacy

- behavior_changed: `False`
- ocr_routing_changed: `False`
- ocr_engine_changed: `False`
- language_detector_behavior_changed: `False`
- classifier_behavior_changed: `False`
- thresholds_changed: `False`
- scoring_changed: `False`
- auto_accept_changed: `False`
- cue_packs_changed: `False`
- cue_expansion_recommended: `False`
- lab_value_parsing_added: `False`
- medication_parsing_added: `False`
- dose_parsing_added: `False`
- ddi_logic_changed: `False`
- clinical_interpretation_added: `False`
- b07_changed: `False`
- route_fix_changed: `False`
- db_schema_changed: `False`
- command_allowlist_changed: `False`
- external_api_changed: `False`
- external_api_used: `False`
- raw_filenames_in_public_reports: `False`
- raw_ocr_text_in_public_reports: `False`
- raw_document_text_in_public_reports: `False`
- private_paths_in_public_reports: `False`
- source_documents_staged: `False`
- private_corpus_files_staged: `False`
- secrets_in_public_reports: `False`
- all_records_remain_review_bound: `True`
- ambiguous_below_threshold_excluded: `True`
- fallback_ran_but_no_family_match_deferred: `True`

No raw filenames, raw OCR text, raw document text, private paths, PHI, or secrets are included. Diagnostic-only, evaluation/reporting changes only. Cue expansion is not recommended for any bucket in this block.

## Secondary signal counts (per subset)

### language_script_visible_detector_unresolved

- table_like_structure_yes: `29`
- numeric_content_medium_or_high: `16`
- section_heading_yes: `3`
- medical_abbreviation_yes: `8`

Raw signal distributions (controlled vocabulary):

- dominant_script_counts: `latin`=31
- language_visibility_status_counts: `latin_visible_language_unknown`=31
- native_text_length_bucket_counts: `long`=1, `medium`=30
- numeric_content_bucket_counts: `low`=15, `medium`=16
- table_like_structure_detected_counts: `no`=2, `yes`=29

### likely_text_layer_issue

- table_like_structure_yes: `10`
- native_text_length_none: `10`
- native_text_length_tiny: `3`
- native_text_length_short: `8`
- alphabetic_low_or_medium: `0`

Raw signal distributions (controlled vocabulary):

- pdf_text_layer_detected_counts: `yes`=21
- image_like_pdf_counts: `no`=21
- native_text_length_bucket_counts: `none`=10, `short`=8, `tiny`=3
- table_like_structure_detected_counts: `no`=11, `yes`=10
- alphabetic_content_bucket_counts: `high`=21

## Why a root-cause classifier and not an implementation

Each priority record is projected to a controlled-vocabulary root-cause label using only privacy-safe metadata already published in the FAMILY-04 per-file table. The next-action bucket names the operational diagnostic lever a follow-up block should investigate, not an implementation change. Runtime behavior is unchanged.

## What this block did not change

- OCR routing logic
- OCR engine
- Language / script detector behavior
- Classifier behavior or cue packs
- Confidence thresholds or scoring
- Auto-accept / review-bound policy
- B07 terminology, ROUTE-FIX, DB schema, command allowlist, external APIs
