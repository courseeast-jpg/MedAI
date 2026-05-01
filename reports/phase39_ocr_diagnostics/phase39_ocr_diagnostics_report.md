# Phase 39 OCR/Layout Diagnostic Reconciliation

- Generated at: `2026-05-01T15:44:30.297753+00:00`
- Source Phase38 batch report: `G:\Codex\2026-04-22-connect-github\reports\phase38_ocr_layout\latest_phase38_batch_validation.json`

## Baselines

- Phase37: total `8`, accepted `2`, review_ocr_quality `6`, empty `0`
- Phase38: total `8`, accepted `2`, review `6`, review_ocr_quality `4`, empty `0`
- Phase38 improved prior review_ocr_quality files: `2`

## Phase39 Diagnostics

- Total files: `8`
- OCR status mismatches: `3`
- review_ocr_quality files: `4`
- Safety regression: `False`
- Status taxonomy changed: `False`

## Mismatch Files

- `Test Results 2.pdf`
- `Test Results 3.pdf`
- `Test Results 6.pdf`

## Per-file Diagnostic Table

| File | OCR band | OCR score | OCR route | Final status | Final reason | Entities | Confidence | Empty | Table/layout warning | Mismatch |
| --- | --- | ---: | --- | --- | --- | ---: | ---: | --- | --- | --- |
| Results 1.pdf | usable_with_review | 0.745 | scanned_or_low_text | accepted | accepted_clean_input | 4 | 0.767 | no | yes | no |
| Results 2.pdf | usable_with_review | 0.735 | scanned_or_low_text | accepted | accepted_clean_input | 3 | 0.693 | no | yes | no |
| Test Results 2.pdf | good | 0.818 | digital_clean_text | review_ocr_quality | table_structure_loss,extraction_low_coverage,classifier_legacy_ocr_flag,legacy_normalized_low_coverage | 6 | 0.7 | no | yes | good_input_but_downstream_ocr_review |
| Test Results 3.pdf | good | 0.811 | digital_clean_text | review_ocr_quality | table_structure_loss,extraction_low_coverage,extraction_low_confidence,safety_gate_low_confidence,classifier_legacy_ocr_flag,legacy_normalized_low_coverage | 4 | 0.63 | no | yes | good_input_but_downstream_ocr_review |
| Test Results 4.pdf | good | 0.816 | scanned_or_low_text | accepted | accepted_clean_input | 4 | 0.693 | no | yes | no |
| Test Results 5.pdf | usable_with_review | 0.796 | digital_clean_text | review_ocr_quality | table_structure_loss,extraction_low_coverage,extraction_low_confidence,safety_gate_low_confidence,classifier_legacy_ocr_flag,legacy_normalized_low_coverage | 3 | 0.63 | no | yes | no |
| Test Results 6.pdf | good | 0.809 | digital_clean_text | review_ocr_quality | table_structure_loss,extraction_low_coverage,extraction_low_confidence,safety_gate_low_confidence,classifier_legacy_ocr_flag,legacy_normalized_low_coverage | 3 | 0.63 | no | yes | good_input_but_downstream_ocr_review |
| Urinalysis, Routine.pdf | usable_with_review | 0.925 | digital_clean_text | accepted | accepted_clean_input | 12 | 0.792 | no | no | no |

## review_ocr_quality Reasons

- `Test Results 2.pdf`: `table_structure_loss,extraction_low_coverage,classifier_legacy_ocr_flag,legacy_normalized_low_coverage`
- `Test Results 3.pdf`: `table_structure_loss,extraction_low_coverage,extraction_low_confidence,safety_gate_low_confidence,classifier_legacy_ocr_flag,legacy_normalized_low_coverage`
- `Test Results 5.pdf`: `table_structure_loss,extraction_low_coverage,extraction_low_confidence,safety_gate_low_confidence,classifier_legacy_ocr_flag,legacy_normalized_low_coverage`
- `Test Results 6.pdf`: `table_structure_loss,extraction_low_coverage,extraction_low_confidence,safety_gate_low_confidence,classifier_legacy_ocr_flag,legacy_normalized_low_coverage`

## Phase40 Recommendation

Keep review_ocr_quality as the safe final status for now, but in Phase40 split the legacy normalized-low-coverage case into an extraction/table-structure diagnostic reason or review_extraction_quality path after downstream coverage metrics are calibrated against lab/table ground truth.
