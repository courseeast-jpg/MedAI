# Phase 10 Audit Report

- Generated at: 2026-04-24T20:33:35.659726+00:00
- Overall passed: True

## Validation Matrix

| Scenario | Passed | Outcome | Validation | Route | Actual | Fallback | Confidence | Band | Error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini_unavailable | True | queued_for_review | accepted | phi3 | phi3 | gemini unavailable | 0.820 | auto_accept | connector_failure |
| claude_unavailable | True | fallback_to_rules | rejected | claude | rules_based | claude_marked_unavailable | 0.450 | reject | claude_unavailable |
| all_external_apis_unavailable | True | written | accepted | spacy | spacy | gemini unavailable | 0.720 | auto_accept | connector_failure |
| ocr_failure | True | rejected | rejected | ocr_validator | ocr_validator | ocr_quality_failure | 0.730 | auto_accept | ocr_failure |
| corrupt_pdf | True | rejected | rejected | pdf_pipeline | pdf_pipeline | corrupt_pdf | 0.000 | reject | corrupt_pdf |
| empty_extraction | True | queued_for_review | rejected | spacy | spacy |  | 0.950 | auto_accept | empty_extraction |
| low_confidence_extraction | True | queued_for_review | rejected | spacy | spacy |  | 0.450 | reject | confidence_below_reject_threshold |
| confidence_below_reject_threshold | True | queued_for_review | rejected | spacy | spacy |  | 0.300 | reject | confidence_below_reject_threshold |
| confidence_in_review_band | True | queued_for_review | needs_review | spacy | spacy |  | 0.600 | review | confidence_below_accept_threshold |
| confidence_above_auto_accept_threshold | True | written | accepted | spacy | spacy |  | 0.900 | auto_accept |  |
| model_route_vs_actual_extractor_mismatch | True | written | accepted | rules_based | rules_based |  | 0.850 | auto_accept |  |

## Audit Completeness

- All required fields present: True
- Failed rows include error_category: True

## Performance

- Average extraction time (ms): 1.919
- p95 extraction time (ms): 2.79
- API call count: 11
- Failed jobs: 6
- Fallback count: 3

## Tier Distribution

- active: 6
- hypothesis: 1
- quarantined: 4
