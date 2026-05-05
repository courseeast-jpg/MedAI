# Phase 62 Table Geometry Header Inference Prototype Diagnostic

- Generated at: `2026-05-04T20:02:44.954115+00:00`
- Source Phase 61 report: `reports\phase61_header_label_inference_diagnostic\phase61_header_label_inference_diagnostic_report.json`
- Subset seed: `20260503`
- Subset size requested: `20`
- Subset size actual: `20`
- geometry_inference_population: `38`
- recoverable_table_candidate_count: `0`
- recoverable_table_candidate_rate_bucket: `very_low_lt_5pct`
- Conclusion: `table_geometry_prototype_assessed`

## Geometry Signal Strength Distribution

| Strength | Count |
| --- | ---: |
| `medium` | 20 |

## Recovery Confidence Band Distribution

| Band | Count |
| --- | ---: |
| `low` | 20 |

## Safe Next Action Distribution

| Action | Count |
| --- | ---: |
| `manual_review_boundary` | 20 |

## Boolean Indicator Distributions (true/false counts)

| Indicator | True | False |
| --- | ---: | ---: |
| `column_alignment_detected` | 20 | 0 |
| `deep_table_block_present` | 0 | 20 |
| `multi_block_structure` | 2 | 18 |
| `recoverable_table_candidate` | 0 | 20 |

## Recommended Phase 63 Target

- Title: `no_code_change_manual_review_boundary`
- production_extractor_should_change_yet: `False`
- recoverable_rate: `very_low_lt_5pct`
- dominant_confidence: `low`

_Insufficient recoverable-geometry signal to justify a prototype. Keep manual review boundary; do not change production extractor._

## Forensic Subset Entries

| safe_file_id | pages | text_len | num_line_ratio | col_count | aligned_pairs | blocks | max_depth | repeated_cols | col_align | deep_block | multi_block | signal | recoverable | confidence | action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `corpus_file_000133` | 4_10 | 5000_20000 | very_low_lt_5pct | high_16_50 | medium_6_15 | very_low_1_2 | very_low_1_2 | zero | yes | no | no | medium | no | low | manual_review_boundary |
| `corpus_file_000215` | 2_3 | 1000_5000 | very_low_lt_5pct | low_3_5 | medium_6_15 | low_3_5 | very_low_1_2 | zero | yes | no | yes | medium | no | low | manual_review_boundary |
| `corpus_file_000221` | 2_3 | 1000_5000 | very_low_lt_5pct | medium_6_15 | medium_6_15 | very_low_1_2 | very_low_1_2 | zero | yes | no | yes | medium | no | low | manual_review_boundary |
| `corpus_file_000247` | 1 | 200_1000 | very_low_lt_5pct | medium_6_15 | medium_6_15 | very_low_1_2 | very_low_1_2 | zero | yes | no | no | medium | no | low | manual_review_boundary |
| `corpus_file_000256` | 1 | 200_1000 | very_low_lt_5pct | medium_6_15 | medium_6_15 | very_low_1_2 | very_low_1_2 | zero | yes | no | no | medium | no | low | manual_review_boundary |
| `corpus_file_000260` | 1 | 200_1000 | very_low_lt_5pct | medium_6_15 | medium_6_15 | very_low_1_2 | very_low_1_2 | zero | yes | no | no | medium | no | low | manual_review_boundary |
| `corpus_file_000269` | 1 | 200_1000 | very_low_lt_5pct | medium_6_15 | medium_6_15 | very_low_1_2 | very_low_1_2 | zero | yes | no | no | medium | no | low | manual_review_boundary |
| `corpus_file_000293` | 1 | 200_1000 | very_low_lt_5pct | medium_6_15 | medium_6_15 | very_low_1_2 | very_low_1_2 | zero | yes | no | no | medium | no | low | manual_review_boundary |
| `corpus_file_000334` | 1 | 200_1000 | very_low_lt_5pct | medium_6_15 | medium_6_15 | very_low_1_2 | very_low_1_2 | zero | yes | no | no | medium | no | low | manual_review_boundary |
| `corpus_file_000354` | 1 | 200_1000 | very_low_lt_5pct | medium_6_15 | medium_6_15 | very_low_1_2 | very_low_1_2 | zero | yes | no | no | medium | no | low | manual_review_boundary |
| `corpus_file_000372` | 1 | 200_1000 | very_low_lt_5pct | low_3_5 | medium_6_15 | very_low_1_2 | very_low_1_2 | zero | yes | no | no | medium | no | low | manual_review_boundary |
| `corpus_file_000386` | 1 | 200_1000 | very_low_lt_5pct | low_3_5 | medium_6_15 | very_low_1_2 | very_low_1_2 | zero | yes | no | no | medium | no | low | manual_review_boundary |
| `corpus_file_000401` | 1 | 200_1000 | very_low_lt_5pct | medium_6_15 | medium_6_15 | very_low_1_2 | very_low_1_2 | zero | yes | no | no | medium | no | low | manual_review_boundary |
| `corpus_file_000404` | 1 | 200_1000 | very_low_lt_5pct | low_3_5 | medium_6_15 | very_low_1_2 | very_low_1_2 | zero | yes | no | no | medium | no | low | manual_review_boundary |
| `corpus_file_000419` | 1 | 200_1000 | very_low_lt_5pct | low_3_5 | medium_6_15 | very_low_1_2 | very_low_1_2 | zero | yes | no | no | medium | no | low | manual_review_boundary |
| `corpus_file_000422` | 1 | 200_1000 | very_low_lt_5pct | low_3_5 | medium_6_15 | very_low_1_2 | very_low_1_2 | zero | yes | no | no | medium | no | low | manual_review_boundary |
| `corpus_file_000473` | 1 | 200_1000 | very_low_lt_5pct | low_3_5 | medium_6_15 | zero | zero | zero | yes | no | no | medium | no | low | manual_review_boundary |
| `corpus_file_000476` | 1 | 200_1000 | very_low_lt_5pct | low_3_5 | medium_6_15 | zero | zero | zero | yes | no | no | medium | no | low | manual_review_boundary |
| `corpus_file_000513` | 2_3 | 1000_5000 | very_low_lt_5pct | low_3_5 | medium_6_15 | very_low_1_2 | very_low_1_2 | zero | yes | no | no | medium | no | low | manual_review_boundary |
| `corpus_file_000515` | 1 | 200_1000 | very_low_lt_5pct | low_3_5 | medium_6_15 | zero | zero | zero | yes | no | no | medium | no | low | manual_review_boundary |

## Privacy Safety

- uses_safe_ids_only: `True`
- raw_filenames_present_in_output: `False`
- raw_paths_present_in_output: `False`
- extracted_text_present_in_output: `False`
- ocr_text_present_in_output: `False`
- table_rows_or_inferred_labels_in_output: `False`
- phi_present_in_output: `False`
- external_api_used: `False`

