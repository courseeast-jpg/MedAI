# Phase 60 Text Extraction Gap Vocabulary Coverage Diagnostic

- Generated at: `2026-05-04T02:56:31.609202+00:00`
- Source Phase 57A report: `reports\phase57_full_corpus_inventory_audit\phase57_full_corpus_inventory_audit_report.json`
- Source Phase 59 report: `reports\phase59_empty_extraction_forensics\phase59_empty_extraction_forensics_report.json`
- Subset seed: `20260503`
- Subset size requested: `60`
- Subset size actual: `60`
- Stratification (file_type → count): `{'pdf': 60}`
- Conclusion: `text_extraction_gap_diagnosed`

## Likely Document Class Distribution

| Class | Count |
| --- | ---: |
| `unknown` | 48 |
| `lab_report` | 8 |
| `prescription_or_medication` | 3 |
| `radiology_or_imaging` | 1 |

- Dominant document class: `unknown`

## Likely Gap Type Distribution

| Gap | Count |
| --- | ---: |
| `numeric_table_without_labels` | 38 |
| `lab_table_vocabulary_gap` | 7 |
| `unknown` | 7 |
| `document_class_not_supported` | 3 |
| `tokenization_or_layout_issue` | 3 |
| `imaging_report_vocabulary_gap` | 1 |
| `medical_vocabulary_gap` | 1 |

- Dominant gap type: `numeric_table_without_labels`

## Coverage Table (bucket → count)

### `medical_token_hit_count_bucket`

| Bucket | Count |
| --- | ---: |
| `zero` | 48 |
| `very_high_gt_50` | 6 |
| `medium_6_15` | 3 |
| `very_low_1_2` | 2 |
| `high_16_50` | 1 |

### `lab_token_hit_count_bucket`

| Bucket | Count |
| --- | ---: |
| `zero` | 51 |
| `very_high_gt_50` | 5 |
| `high_16_50` | 1 |
| `low_3_5` | 1 |
| `medium_6_15` | 1 |
| `very_low_1_2` | 1 |

### `unit_pattern_count_bucket`

| Bucket | Count |
| --- | ---: |
| `zero` | 55 |
| `very_high_gt_50` | 4 |
| `medium_6_15` | 1 |

### `numeric_density_bucket`

| Bucket | Count |
| --- | ---: |
| `high_30_50pct` | 23 |
| `very_high_gt_50pct` | 19 |
| `low_5_15pct` | 7 |
| `medium_15_30pct` | 6 |
| `very_low_lt_5pct` | 5 |

### `table_like_line_count_bucket`

| Bucket | Count |
| --- | ---: |
| `zero` | 55 |
| `low_3_5` | 4 |
| `very_low_1_2` | 1 |

## Recommended Phase 61 Target

- Title: `lab_table_vocabulary_expansion_diagnostic`
- production_extractor_should_change_yet: `False`

_Numeric-heavy table-like content without labels. Phase 61 should evaluate header/label inference as a diagnostic, not an extractor change._

## Forensic Subset Entries

| safe_file_id | file_type | size | pages | text_len | medical_hits | lab_hits | unit_patterns | numeric_density | table_like | doc_class | gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `corpus_file_000004` | pdf | 100KB_1MB | 4_10 | 5000_20000 | very_high_gt_50 | very_high_gt_50 | very_high_gt_50 | high_30_50pct | low_3_5 | `lab_report` | `lab_table_vocabulary_gap` |
| `corpus_file_000024` | pdf | 100KB_1MB | 4_10 | 5000_20000 | very_high_gt_50 | very_high_gt_50 | very_high_gt_50 | high_30_50pct | low_3_5 | `lab_report` | `lab_table_vocabulary_gap` |
| `corpus_file_000085` | pdf | 1_10MB | 2_3 | 200_1000 | zero | zero | zero | medium_15_30pct | zero | `unknown` | `unknown` |
| `corpus_file_000127` | pdf | 1_10MB | 2_3 | 1000_5000 | medium_6_15 | low_3_5 | zero | low_5_15pct | zero | `lab_report` | `lab_table_vocabulary_gap` |
| `corpus_file_000215` | pdf | 100KB_1MB | 2_3 | 1000_5000 | zero | zero | zero | high_30_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000221` | pdf | 100KB_1MB | 2_3 | 1000_5000 | zero | zero | zero | high_30_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000247` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | very_high_gt_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000260` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | very_high_gt_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000282` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | very_high_gt_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000334` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | very_high_gt_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000359` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | very_high_gt_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000387` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | high_30_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000404` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | high_30_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000422` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | high_30_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000423` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | high_30_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000449` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | high_30_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000458` | pdf | 100KB_1MB | 2_3 | 1000_5000 | zero | zero | zero | low_5_15pct | zero | `unknown` | `unknown` |
| `corpus_file_000473` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | high_30_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000483` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | low_5_15pct | zero | `unknown` | `unknown` |
| `corpus_file_000485` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | medium_15_30pct | zero | `unknown` | `unknown` |
| `corpus_file_000506` | pdf | 100KB_1MB | 4_10 | 5000_20000 | very_high_gt_50 | very_high_gt_50 | very_high_gt_50 | high_30_50pct | low_3_5 | `lab_report` | `lab_table_vocabulary_gap` |
| `corpus_file_000507` | pdf | 100KB_1MB | 2_3 | 1000_5000 | zero | zero | zero | high_30_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000548` | pdf | 1_10MB | 1 | 1000_5000 | medium_6_15 | zero | zero | low_5_15pct | zero | `prescription_or_medication` | `document_class_not_supported` |
| `corpus_file_000551` | pdf | 1_10MB | 4_10 | 5000_20000 | very_high_gt_50 | very_high_gt_50 | medium_6_15 | medium_15_30pct | zero | `lab_report` | `lab_table_vocabulary_gap` |
| `corpus_file_000591` | pdf | 100KB_1MB | 2_3 | 1000_5000 | zero | zero | zero | low_5_15pct | zero | `unknown` | `unknown` |
| `corpus_file_000011` | pdf | 100KB_1MB | 2_3 | 1000_5000 | zero | zero | zero | medium_15_30pct | very_low_1_2 | `unknown` | `unknown` |
| `corpus_file_000030` | pdf | 100KB_1MB | 4_10 | 5000_20000 | very_high_gt_50 | very_high_gt_50 | very_high_gt_50 | high_30_50pct | low_3_5 | `lab_report` | `lab_table_vocabulary_gap` |
| `corpus_file_000091` | pdf | 100KB_1MB | 2_3 | 1000_5000 | medium_6_15 | zero | zero | very_low_lt_5pct | zero | `prescription_or_medication` | `document_class_not_supported` |
| `corpus_file_000099` | pdf | 100KB_1MB | 1 | 200_1000 | zero | zero | zero | medium_15_30pct | zero | `unknown` | `unknown` |
| `corpus_file_000100` | pdf | 1_10MB | 2_3 | zero | zero | zero | zero | very_low_lt_5pct | zero | `unknown` | `tokenization_or_layout_issue` |
| `corpus_file_000133` | pdf | 100KB_1MB | 4_10 | 5000_20000 | zero | zero | zero | very_high_gt_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000136` | pdf | 100KB_1MB | 4_10 | 5000_20000 | very_high_gt_50 | high_16_50 | zero | low_5_15pct | zero | `prescription_or_medication` | `document_class_not_supported` |
| `corpus_file_000223` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | high_30_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000233` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | very_high_gt_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000249` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | very_high_gt_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000253` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | very_high_gt_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000256` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | very_high_gt_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000268` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | very_high_gt_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000269` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | very_high_gt_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000270` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | very_high_gt_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000293` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | very_high_gt_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000343` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | very_high_gt_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000346` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | very_high_gt_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000354` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | very_high_gt_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000372` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | very_high_gt_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000385` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | high_30_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000386` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | high_30_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000401` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | high_30_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000419` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | high_30_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000439` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | high_30_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000440` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | very_high_gt_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000476` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | high_30_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000488` | pdf | 100KB_1MB | 2_3 | 50_200 | zero | zero | zero | very_low_lt_5pct | zero | `unknown` | `tokenization_or_layout_issue` |
| `corpus_file_000499` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | high_30_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000500` | pdf | 100KB_1MB | 2_3 | 50_200 | zero | zero | zero | very_low_lt_5pct | zero | `unknown` | `tokenization_or_layout_issue` |
| `corpus_file_000513` | pdf | 10_100KB | 2_3 | 1000_5000 | zero | zero | zero | high_30_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000515` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | high_30_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000534` | pdf | 1_10MB | 2_3 | 1000_5000 | very_low_1_2 | very_low_1_2 | zero | low_5_15pct | zero | `lab_report` | `medical_vocabulary_gap` |
| `corpus_file_000536` | pdf | 1_10MB | 1 | 200_1000 | very_low_1_2 | zero | zero | medium_15_30pct | zero | `radiology_or_imaging` | `imaging_report_vocabulary_gap` |
| `corpus_file_000545` | pdf | 1_10MB | 4_10 | 5000_20000 | high_16_50 | medium_6_15 | zero | very_low_lt_5pct | zero | `lab_report` | `lab_table_vocabulary_gap` |

## Privacy Safety

- uses_safe_ids_only: `True`
- raw_filenames_present_in_output: `False`
- raw_paths_present_in_output: `False`
- extracted_text_present_in_output: `False`
- ocr_text_present_in_output: `False`
- phi_present_in_output: `False`
- external_api_used: `False`

