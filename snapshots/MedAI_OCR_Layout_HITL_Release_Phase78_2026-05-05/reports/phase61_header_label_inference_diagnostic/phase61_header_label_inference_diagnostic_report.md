# Phase 61 Header/Label Inference Diagnostic for Numeric Table Gaps

- Generated at: `2026-05-04T15:01:09.885260+00:00`
- Source Phase 60 report: `reports\phase60_text_extraction_gap_diagnostic\phase60_text_extraction_gap_diagnostic_report.json`
- Subset seed: `20260503`
- Subset size requested: `40`
- Subset size actual: `38`
- numeric_table_without_labels population: `38`
- Conclusion: `header_label_inference_diagnosed`

## Recommended Strategy Distribution

| Strategy | Count |
| --- | ---: |
| `table_geometry_header_inference` | 38 |

- Dominant recommended strategy: `table_geometry_header_inference`

## Confidence Band Distribution

| Band | Count |
| --- | ---: |
| `high` | 38 |

## Boolean Indicator Distributions (true/false counts)

| Indicator | True | False |
| --- | ---: | ---: |
| `likely_table_present` | 38 | 0 |
| `likely_header_missing` | 38 | 0 |
| `likely_header_fragmented` | 0 | 38 |
| `likely_non_english_labels` | 0 | 38 |
| `likely_cyrillic_or_mixed_script_labels` | 0 | 38 |
| `likely_units_without_analyte_names` | 0 | 38 |
| `likely_analyte_names_without_units` | 0 | 38 |
| `inferable_by_neighbor_lines` | 0 | 38 |
| `inferable_by_table_geometry` | 38 | 0 |
| `inferable_by_generic_lab_units` | 0 | 38 |
| `inferable_by_multilingual_label_map` | 0 | 38 |
| `repeated_numeric_column_pattern` | 0 | 38 |

## Recommended Phase 62 Target

- Title: `phase62_table_geometry_header_inference_prototype`
- production_extractor_should_change_yet: `False`
- dominant_strategy: `table_geometry_header_inference`

_Most files have aligned numeric columns without recognised headers. Phase 62 should evaluate a NARROW geometry diagnostic prototype only — no production extractor change._

## Forensic Subset Entries

| safe_file_id | pages | text_len | numeric_density | unit_patterns | strategy | confidence | table | header_missing | non_english | inferable_neighbor | inferable_geom | inferable_units | inferable_multilingual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `corpus_file_000133` | 4_10 | 5000_20000 | very_high_gt_50pct | zero | `table_geometry_header_inference` | high | yes | yes | no | no | yes | no | no |
| `corpus_file_000215` | 2_3 | 1000_5000 | high_30_50pct | zero | `table_geometry_header_inference` | high | yes | yes | no | no | yes | no | no |
| `corpus_file_000221` | 2_3 | 1000_5000 | high_30_50pct | zero | `table_geometry_header_inference` | high | yes | yes | no | no | yes | no | no |
| `corpus_file_000223` | 1 | 200_1000 | high_30_50pct | zero | `table_geometry_header_inference` | high | yes | yes | no | no | yes | no | no |
| `corpus_file_000233` | 1 | 200_1000 | very_high_gt_50pct | zero | `table_geometry_header_inference` | high | yes | yes | no | no | yes | no | no |
| `corpus_file_000247` | 1 | 200_1000 | very_high_gt_50pct | zero | `table_geometry_header_inference` | high | yes | yes | no | no | yes | no | no |
| `corpus_file_000249` | 1 | 200_1000 | very_high_gt_50pct | zero | `table_geometry_header_inference` | high | yes | yes | no | no | yes | no | no |
| `corpus_file_000253` | 1 | 200_1000 | very_high_gt_50pct | zero | `table_geometry_header_inference` | high | yes | yes | no | no | yes | no | no |
| `corpus_file_000256` | 1 | 200_1000 | very_high_gt_50pct | zero | `table_geometry_header_inference` | high | yes | yes | no | no | yes | no | no |
| `corpus_file_000260` | 1 | 200_1000 | very_high_gt_50pct | zero | `table_geometry_header_inference` | high | yes | yes | no | no | yes | no | no |
| `corpus_file_000268` | 1 | 200_1000 | very_high_gt_50pct | zero | `table_geometry_header_inference` | high | yes | yes | no | no | yes | no | no |
| `corpus_file_000269` | 1 | 200_1000 | very_high_gt_50pct | zero | `table_geometry_header_inference` | high | yes | yes | no | no | yes | no | no |
| `corpus_file_000270` | 1 | 200_1000 | very_high_gt_50pct | zero | `table_geometry_header_inference` | high | yes | yes | no | no | yes | no | no |
| `corpus_file_000282` | 1 | 200_1000 | very_high_gt_50pct | zero | `table_geometry_header_inference` | high | yes | yes | no | no | yes | no | no |
| `corpus_file_000293` | 1 | 200_1000 | very_high_gt_50pct | zero | `table_geometry_header_inference` | high | yes | yes | no | no | yes | no | no |
| `corpus_file_000334` | 1 | 200_1000 | very_high_gt_50pct | zero | `table_geometry_header_inference` | high | yes | yes | no | no | yes | no | no |
| `corpus_file_000343` | 1 | 200_1000 | very_high_gt_50pct | zero | `table_geometry_header_inference` | high | yes | yes | no | no | yes | no | no |
| `corpus_file_000346` | 1 | 200_1000 | very_high_gt_50pct | zero | `table_geometry_header_inference` | high | yes | yes | no | no | yes | no | no |
| `corpus_file_000354` | 1 | 200_1000 | very_high_gt_50pct | zero | `table_geometry_header_inference` | high | yes | yes | no | no | yes | no | no |
| `corpus_file_000359` | 1 | 200_1000 | very_high_gt_50pct | zero | `table_geometry_header_inference` | high | yes | yes | no | no | yes | no | no |
| `corpus_file_000372` | 1 | 200_1000 | very_high_gt_50pct | zero | `table_geometry_header_inference` | high | yes | yes | no | no | yes | no | no |
| `corpus_file_000385` | 1 | 200_1000 | high_30_50pct | zero | `table_geometry_header_inference` | high | yes | yes | no | no | yes | no | no |
| `corpus_file_000386` | 1 | 200_1000 | high_30_50pct | zero | `table_geometry_header_inference` | high | yes | yes | no | no | yes | no | no |
| `corpus_file_000387` | 1 | 200_1000 | high_30_50pct | zero | `table_geometry_header_inference` | high | yes | yes | no | no | yes | no | no |
| `corpus_file_000401` | 1 | 200_1000 | high_30_50pct | zero | `table_geometry_header_inference` | high | yes | yes | no | no | yes | no | no |
| `corpus_file_000404` | 1 | 200_1000 | high_30_50pct | zero | `table_geometry_header_inference` | high | yes | yes | no | no | yes | no | no |
| `corpus_file_000419` | 1 | 200_1000 | high_30_50pct | zero | `table_geometry_header_inference` | high | yes | yes | no | no | yes | no | no |
| `corpus_file_000422` | 1 | 200_1000 | high_30_50pct | zero | `table_geometry_header_inference` | high | yes | yes | no | no | yes | no | no |
| `corpus_file_000423` | 1 | 200_1000 | high_30_50pct | zero | `table_geometry_header_inference` | high | yes | yes | no | no | yes | no | no |
| `corpus_file_000439` | 1 | 200_1000 | high_30_50pct | zero | `table_geometry_header_inference` | high | yes | yes | no | no | yes | no | no |
| `corpus_file_000440` | 1 | 200_1000 | very_high_gt_50pct | zero | `table_geometry_header_inference` | high | yes | yes | no | no | yes | no | no |
| `corpus_file_000449` | 1 | 200_1000 | high_30_50pct | zero | `table_geometry_header_inference` | high | yes | yes | no | no | yes | no | no |
| `corpus_file_000473` | 1 | 200_1000 | high_30_50pct | zero | `table_geometry_header_inference` | high | yes | yes | no | no | yes | no | no |
| `corpus_file_000476` | 1 | 200_1000 | high_30_50pct | zero | `table_geometry_header_inference` | high | yes | yes | no | no | yes | no | no |
| `corpus_file_000499` | 1 | 200_1000 | high_30_50pct | zero | `table_geometry_header_inference` | high | yes | yes | no | no | yes | no | no |
| `corpus_file_000507` | 2_3 | 1000_5000 | high_30_50pct | zero | `table_geometry_header_inference` | high | yes | yes | no | no | yes | no | no |
| `corpus_file_000513` | 2_3 | 1000_5000 | high_30_50pct | zero | `table_geometry_header_inference` | high | yes | yes | no | no | yes | no | no |
| `corpus_file_000515` | 1 | 200_1000 | high_30_50pct | zero | `table_geometry_header_inference` | high | yes | yes | no | no | yes | no | no |

## Privacy Safety

- uses_safe_ids_only: `True`
- raw_filenames_present_in_output: `False`
- raw_paths_present_in_output: `False`
- extracted_text_present_in_output: `False`
- ocr_text_present_in_output: `False`
- table_rows_or_labels_in_output: `False`
- phi_present_in_output: `False`
- external_api_used: `False`

