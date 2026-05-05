# Phase 60 Text Extraction Gap Vocabulary Coverage Diagnostic

- Generated at: `2026-05-04T15:02:42.180854+00:00`
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
| `unknown` | 44 |
| `lab_report` | 10 |
| `prescription_or_medication` | 3 |
| `admin_or_billing` | 2 |
| `radiology_or_imaging` | 1 |

- Dominant document class: `unknown`

## Likely Gap Type Distribution

| Gap | Count |
| --- | ---: |
| `numeric_table_without_labels` | 35 |
| `lab_table_vocabulary_gap` | 10 |
| `unknown` | 9 |
| `tokenization_or_layout_issue` | 3 |
| `document_class_not_supported` | 2 |
| `imaging_report_vocabulary_gap` | 1 |

- Dominant gap type: `numeric_table_without_labels`

## Coverage Table (bucket → count)

### `medical_token_hit_count_bucket`

| Bucket | Count |
| --- | ---: |
| `zero` | 46 |
| `very_high_gt_50` | 6 |
| `medium_6_15` | 4 |
| `high_16_50` | 2 |
| `low_3_5` | 1 |
| `very_low_1_2` | 1 |

### `lab_token_hit_count_bucket`

| Bucket | Count |
| --- | ---: |
| `zero` | 48 |
| `very_high_gt_50` | 6 |
| `low_3_5` | 3 |
| `medium_6_15` | 2 |
| `very_low_1_2` | 1 |

### `unit_pattern_count_bucket`

| Bucket | Count |
| --- | ---: |
| `zero` | 53 |
| `very_high_gt_50` | 5 |
| `medium_6_15` | 2 |

### `numeric_density_bucket`

| Bucket | Count |
| --- | ---: |
| `high_30_50pct` | 24 |
| `very_high_gt_50pct` | 19 |
| `medium_15_30pct` | 8 |
| `low_5_15pct` | 6 |
| `very_low_lt_5pct` | 3 |

### `table_like_line_count_bucket`

| Bucket | Count |
| --- | ---: |
| `zero` | 55 |
| `low_3_5` | 5 |

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
| `corpus_file_000513` | pdf | 10_100KB | 2_3 | 1000_5000 | zero | zero | zero | high_30_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000551` | pdf | 1_10MB | 4_10 | 5000_20000 | very_high_gt_50 | very_high_gt_50 | medium_6_15 | medium_15_30pct | zero | `lab_report` | `lab_table_vocabulary_gap` |
| `corpus_file_000554` | pdf | 1_10MB | 4_10 | 5000_20000 | medium_6_15 | medium_6_15 | zero | medium_15_30pct | zero | `lab_report` | `lab_table_vocabulary_gap` |
| `corpus_file_000592` | pdf | 100KB_1MB | 2_3 | 1000_5000 | medium_6_15 | very_low_1_2 | zero | medium_15_30pct | zero | `prescription_or_medication` | `document_class_not_supported` |
| `corpus_file_000599` | pdf | 1_10MB | 1 | 200_1000 | zero | zero | zero | low_5_15pct | zero | `unknown` | `unknown` |
| `corpus_file_000013` | pdf | 100KB_1MB | 4_10 | 5000_20000 | very_high_gt_50 | very_high_gt_50 | very_high_gt_50 | high_30_50pct | low_3_5 | `lab_report` | `lab_table_vocabulary_gap` |
| `corpus_file_000037` | pdf | 100KB_1MB | 4_10 | 5000_20000 | very_high_gt_50 | very_high_gt_50 | very_high_gt_50 | high_30_50pct | low_3_5 | `lab_report` | `lab_table_vocabulary_gap` |
| `corpus_file_000092` | pdf | 10_100KB | 4_10 | 5000_20000 | high_16_50 | low_3_5 | zero | medium_15_30pct | zero | `radiology_or_imaging` | `imaging_report_vocabulary_gap` |
| `corpus_file_000100` | pdf | 1_10MB | 2_3 | zero | zero | zero | zero | very_low_lt_5pct | zero | `unknown` | `tokenization_or_layout_issue` |
| `corpus_file_000101` | pdf | 100KB_1MB | 2_3 | 1000_5000 | zero | zero | zero | medium_15_30pct | zero | `admin_or_billing` | `unknown` |
| `corpus_file_000135` | pdf | 100KB_1MB | 1 | 1000_5000 | low_3_5 | low_3_5 | zero | low_5_15pct | zero | `lab_report` | `lab_table_vocabulary_gap` |
| `corpus_file_000224` | pdf | 10_100KB | 2_3 | 1000_5000 | high_16_50 | medium_6_15 | medium_6_15 | high_30_50pct | zero | `lab_report` | `lab_table_vocabulary_gap` |
| `corpus_file_000234` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | very_high_gt_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000250` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | very_high_gt_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000257` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | very_high_gt_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000269` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | very_high_gt_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000270` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | very_high_gt_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000271` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | very_high_gt_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000294` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | very_high_gt_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000344` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | very_high_gt_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000347` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | very_high_gt_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000355` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | very_high_gt_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000373` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | very_high_gt_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000386` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | high_30_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000388` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | very_high_gt_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000402` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | high_30_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000420` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | high_30_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000440` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | very_high_gt_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000441` | pdf | 10_100KB | 1 | 200_1000 | very_low_1_2 | zero | zero | very_high_gt_50pct | zero | `prescription_or_medication` | `unknown` |
| `corpus_file_000477` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | high_30_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000489` | pdf | 100KB_1MB | 2_3 | 1000_5000 | zero | zero | zero | high_30_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000500` | pdf | 100KB_1MB | 2_3 | 50_200 | zero | zero | zero | very_low_lt_5pct | zero | `unknown` | `tokenization_or_layout_issue` |
| `corpus_file_000501` | pdf | 100KB_1MB | 1 | 200_1000 | zero | zero | zero | high_30_50pct | zero | `admin_or_billing` | `unknown` |
| `corpus_file_000515` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | high_30_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000517` | pdf | 10_100KB | 1 | 200_1000 | zero | zero | zero | high_30_50pct | zero | `unknown` | `numeric_table_without_labels` |
| `corpus_file_000538` | pdf | 1_10MB | 1 | 1000_5000 | zero | zero | zero | medium_15_30pct | zero | `unknown` | `unknown` |
| `corpus_file_000541` | pdf | 1_10MB | 2_3 | zero | zero | zero | zero | very_low_lt_5pct | zero | `unknown` | `tokenization_or_layout_issue` |
| `corpus_file_000548` | pdf | 1_10MB | 1 | 1000_5000 | medium_6_15 | zero | zero | low_5_15pct | zero | `prescription_or_medication` | `document_class_not_supported` |

## Privacy Safety

- uses_safe_ids_only: `True`
- raw_filenames_present_in_output: `False`
- raw_paths_present_in_output: `False`
- extracted_text_present_in_output: `False`
- ocr_text_present_in_output: `False`
- phi_present_in_output: `False`
- external_api_used: `False`

