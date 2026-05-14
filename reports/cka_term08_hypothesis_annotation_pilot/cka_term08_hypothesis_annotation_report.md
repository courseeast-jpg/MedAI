# CKA-TERM-08 Hypothesis-Only Coding Annotation Pilot Report

Conclusion: `cka_term08_hypothesis_annotation_ready`

## Feature Flag
- Feature flag: `MEDAI_TERMINOLOGY_HYPOTHESIS_ANNOTATION`
- Default enabled: `False`
- Disabled mode preserves current behavior: `True`

## Validation Cases
- Total: `4`
- Passed: `4`
- Failed: `0`
- feature_flag_off: passed=`True`, status=`disabled`, candidate_code_count=`0`
- exact_lookup_hypothesis_only: passed=`True`, status=`exact`, candidate_code_count=`1`
- unknown_unmapped_no_hallucination: passed=`True`, status=`unmapped`, candidate_code_count=`0`
- ambiguous_manual_review: passed=`True`, status=`ambiguous`, candidate_code_count=`2`

## Safety
- Writes active fact: `False`
- Clears DDI status: `False`
- Promotes hypothesis: `False`
- External API used: `False`
- Clinical recommendations generated: `False`
- Dosing advice generated: `False`
- B07 behavior changed: `False`
- OCR/extractor/safety gates changed: `False`
- Privacy report clean: `True`

## Next Action
Create PARK-08 snapshot before any B07-TERM opt-in integration.
