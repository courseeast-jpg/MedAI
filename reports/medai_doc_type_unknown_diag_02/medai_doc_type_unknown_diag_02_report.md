# MEDAI-DOC-TYPE-UNKNOWN-DIAG-02 - OCR/Text Visibility & Fallback Shape Audit

- branch: `clinical-knowledge-architecture`
- HEAD commit (short): `455147097916`
- PARK-19 baseline commit (short): `ac466e0f9ab8`
- public_report_commit_hash_policy: `short_hashes_only`
- upstream diagnostic: `reports/medai_doc_type_unknown_diag_01/(public diagnostic)`
- source report: `reports/medai_doc_type_family_04_larger_slice_validation/(public anonymized batch-eval per-file table)`
- total Unknown analyzed: `107`
- generated_at: `2026-05-18T18:01:27.640614+00:00`

## Bucket counts

- insufficient_text_visibility: `75`
- fallback_ran_but_no_family_match: `17`
- ambiguous_below_threshold: `15`

## Section A - OCR / text-visibility eligibility

- count: `75`
- ocr_visibility_breakdown:
  - likely_text_layer_issue: `21`
  - likely_ocr_eligibility_issue: `0`
  - image_like_but_ocr_not_routed: `11`
  - no_text_layer: `11`
  - language_script_visible_detector_unresolved: `31`
  - fallback_eligible_but_not_triggered: `0`
  - extraction_error: `0`
  - non_actionable_leave_manual_review: `1`
- block_justified: `True`
- justification: 74 of 75 records show actionable text-visibility issues. A future OCR / text-visibility evaluation-only block is justified, prioritizing language_script_visible_detector_unresolved and likely_text_layer_issue. No runtime change required by this block.

## Section B - fallback shape audit

- count: `17`
- shape_audit_counts:
  - possible_lab_shape_without_language_cues: `2`
  - possible_imaging_shape_without_language_cues: `1`
  - possible_treatment_shape_without_language_cues: `11`
  - generic_form_shapes_only: `1`
  - likely_nonmedical_or_header_noise: `0`
  - no_known_family_shapes: `0`
  - needs_manual_review: `2`
- block_justified: `True`
- justification: 14 of 17 fallback-bucket records surface a recognizable medical shape without language cues. A narrow cue-audit / cue-coverage review block (evaluation-only, no cue expansion) is justified. The largest sub-category should steer the audit focus.

## Section C - ambiguous_below_threshold (summary only)

- count: `15`
- cue_expansion_recommended: `False`
- note: Higher false-positive risk. Reported as a summary count only per UNKNOWN-DIAG-02 scope. All records remain review-bound; cue expansion must not be driven from this bucket.

## Overall recommendation

- recommendation: `A_then_B`

Both an OCR / text-visibility evaluation-only block (Section A) and a narrow cue-coverage audit block (Section B) are justified. Recommend sequencing Section A first (larger pool, upstream cause) and then Section B (smaller pool, downstream).

## Safety / Privacy

- behavior_changed: `False`
- ocr_routing_changed: `False`
- ocr_engine_changed: `False`
- classifier_behavior_changed: `False`
- thresholds_changed: `False`
- scoring_changed: `False`
- auto_accept_changed: `False`
- cue_packs_changed: `False`
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

No raw filenames, raw OCR text, raw document text, private paths, PHI, or secrets are included. Diagnostic-only, evaluation/reporting changes only.
