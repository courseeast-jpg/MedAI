# MEDAI-DOC-TYPE-UNKNOWN-DIAG-01 - Unknown Residual Diagnostic

- branch: `clinical-knowledge-architecture`
- HEAD commit (short): `2446d7719a3e`
- PARK-19 baseline commit (short): `ac466e0f9ab8`
- public_report_commit_hash_policy: `short_hashes_only`
- source report: `reports/medai_doc_type_family_04_larger_slice_validation/(public anonymized larger-slice report)`
- total Unknown records analyzed: `107`
- generated_at: `2026-05-18T17:31:04.647172+00:00`

## Bucket counts

- insufficient_text_visibility: `75`
- fallback_ran_but_no_family_match: `17`
- ambiguous_below_threshold: `15`

## insufficient_text_visibility

- count: `75`
- sub-breakdown:
  - image_like_pdf_but_not_routed_to_ocr: `11`
  - language_visibility_unknown: `42`
  - no_text_layer: `11`
  - routing_not_eligible: `1`
  - text_layer_present_but_too_short: `21`
- likely_next_action: `improve OCR or text-visibility coverage; must not expand cue packs`
- anonymized sample IDs (synthetic): `insufficient_text_visibility_001`, `insufficient_text_visibility_002`, `insufficient_text_visibility_003`, `insufficient_text_visibility_004`, `insufficient_text_visibility_005`
- notes:
  - image_like_pdf_but_not_routed_to_ocr indicates a candidate for OCR routing review in a later, dedicated block.
  - language_visibility_unknown is the largest sub-bucket; primary lever is text-layer or OCR coverage, not classifier cue addition.
  - Sub-buckets may overlap (a single file can match more than one reason); the bucket count is authoritative, the sub-breakdown is informational.
  - Aggregate-only. No raw filenames, no raw text, no private paths.

## fallback_ran_but_no_family_match

- count: `17`
- sub-breakdown:
  - fallback_ran_no_family_match_aggregate_only: `17`
- likely_next_action: `audit shape cues (lab-like / imaging-like / treatment-like / generic admin / no-known-shapes) in a follow-up evaluation-only block`
- anonymized sample IDs (synthetic): `fallback_ran_but_no_family_match_001`, `fallback_ran_but_no_family_match_002`, `fallback_ran_but_no_family_match_003`, `fallback_ran_but_no_family_match_004`, `fallback_ran_but_no_family_match_005`
- notes:
  - Aggregate count from the FAMILY-04 public report. Per-shape breakdown for this bucket is not present in the public report; EVAL-05 introduced the shape-audit framework that could be invoked in a follow-up evaluation-only block if operational coverage matters.
  - Failure mode is most plausibly missing family cues on degraded text rather than OCR routing -- the bucket name implies fallback ran.
  - No cue expansion in this block.

## ambiguous_below_threshold

- count: `15`
- sub-breakdown:
  - imaging_report_medication_plan_treatment_plan: `7`
  - administrative_insurance_referral_order: `2`
  - imaging_report_medication_plan: `4`
  - administrative_insurance_medication_plan_referral_order: `1`
  - administrative_insurance_medication_plan: `1`
- likely_next_action: `leave review-bound; must not drive cue expansion from this bucket yet`
- anonymized sample IDs (synthetic): `ambiguous_below_threshold_001`, `ambiguous_below_threshold_002`, `ambiguous_below_threshold_003`, `ambiguous_below_threshold_004`, `ambiguous_below_threshold_005`
- notes:
  - Higher false-positive risk. Reported as summary only per the UNKNOWN-DIAG-01 scope.
  - All ambiguities remain review-bound; no auto-accept allowance.

## Conclusion

Primary diagnostic focus: ocr_or_text_visibility. The dominant Unknown bucket is insufficient_text_visibility (75 of 107 records). The fallback_ran_but_no_family_match bucket (17) is a secondary, evaluation-only candidate for a shape-cue audit. ambiguous_below_threshold (15) remains review-bound and is not optimized in this block.

## Recommendation for next block (optional)

If operational coverage requires it, a follow-up evaluation-only block could (1) audit OCR/text-visibility eligibility for the insufficient_text_visibility sub-buckets, prioritizing language_visibility_unknown and text_layer_present_but_too_short, and (2) re-run the EVAL-05 shape audit specifically over the fallback_ran_but_no_family_match bucket to characterize cue gaps. Otherwise leave all 107 Unknown records review-bound.

## Safety / Privacy

- behavior_changed: `False`
- ocr_routing_changed: `False`
- ocr_engine_changed: `False`
- classifier_behavior_changed: `False`
- thresholds_changed: `False`
- scoring_changed: `False`
- auto_accept_changed: `False`
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

No raw filenames, raw OCR text, raw document text, private paths, PHI, or secrets are included in this report. Diagnostic-only, evaluation/reporting changes only.

## Data source

This block reads exclusively from the existing public report:

- `reports/medai_doc_type_family_04_larger_slice_validation/(public anonymized larger-slice report)`

The source is the FAMILY-04 anonymized larger-slice batch evaluation (507 supported files). No corpus was re-read, no source documents were opened, and no external API was invoked.

## Why insufficient_text_visibility is the primary focus

Its dominant sub-buckets - language_visibility_unknown and text_layer_present_but_too_short - are upstream of the classifier. Adding cues would not materially recover these records because the evaluator never had enough text-shape signal to score against any family. The correct lever is OCR routing coverage or improved native text-layer extraction, both of which are out of scope for this block.

## Why ambiguous_below_threshold is summary-only

Each candidate set involves at least one family with higher false-
positive risk if cues are loosened. The block deliberately reports the count and candidate-set distribution but does NOT propose cue expansion. All 15 records remain review-bound.

## What this block did not change

- OCR routing logic
- OCR engine
- Classifier behavior or cue packs
- Confidence thresholds or scoring
- Auto-accept / review-bound policy
- B07 terminology, ROUTE-FIX, DB schema, command allowlist, external APIs
