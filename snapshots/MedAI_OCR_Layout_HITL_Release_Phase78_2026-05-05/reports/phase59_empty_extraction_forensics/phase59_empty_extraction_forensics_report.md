# Phase 59 Empty Extraction Forensic Subset Audit

- Generated at: `2026-05-04T15:02:39.101660+00:00`
- Source Phase 57A report: `reports\phase57_full_corpus_inventory_audit\phase57_full_corpus_inventory_audit_report.json`
- Subset seed: `20260503`
- Subset size requested: `30`
- Subset size actual: `30`
- Empty extraction population: `436`
- Stratification (file_type → count): `{'pdf': 28, 'txt': 1, 'image': 1}`
- Conclusion: `forensic_subset_audited`

## Root Cause Bucket Distribution

| Bucket | Count | Percent |
| --- | ---: | ---: |
| `pdf_text_extraction_gap` | 27 | 90.00% |
| `blank_or_near_blank` | 1 | 3.33% |
| `likely_non_medical_or_admin` | 1 | 3.33% |
| `ocr_ran_but_low_text` | 1 | 3.33% |
| `embedded_or_portfolio_pdf` | 0 | 0.00% |
| `image_only_pdf_needs_ocr` | 0 | 0.00% |
| `pipeline_bug_suspected` | 0 | 0.00% |
| `unknown_needs_manual_review` | 0 | 0.00% |
| `unsupported_or_malformed_structure` | 0 | 0.00% |

- Dominant root cause bucket: `pdf_text_extraction_gap`

## Recommended Phase 60 Target

- Title: `phase60_text_extraction_gap_diagnostic`

_Text is present in the PDF but downstream extraction returns nothing. Phase 60 should run a vocabulary / entity-coverage diagnostic against a subset of these files BEFORE any parser tuning._

## Forensic Subset Entries

| safe_file_id | file_type | extension | size_bucket | OCR | doc_type | pages | text_len | embedded? | bucket |
| --- | --- | --- | --- | --- | --- | ---: | --- | --- | --- |
| `corpus_file_000004` | pdf | .pdf | 100KB_1MB | good | digital_pdf | 6 | gt_5000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000025` | pdf | .pdf | 100KB_1MB | good | digital_pdf | 6 | gt_5000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000087` | pdf | .pdf | 100KB_1MB | good | digital_pdf | 2 | 1000_5000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000097` | pdf | .pdf | 100KB_1MB | usable_with_review | mixed_pdf | 2 | 200_1000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000128` | pdf | .pdf | 10_100KB | good | mixed_pdf | 1 | 200_1000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000161` | txt | .txt | lt_1KB | unknown | text_file |  | 1_50 | no | `blank_or_near_blank` |
| `corpus_file_000206` | image | .jpg | 100KB_1MB | poor_ocr | image_ocr |  |  | no | `ocr_ran_but_low_text` |
| `corpus_file_000216` | pdf | .pdf | 1_10MB | good | mixed_pdf | 3 | gt_5000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000222` | pdf | .pdf | 10_100KB | good | mixed_pdf | 2 | 200_1000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000248` | pdf | .pdf | 10_100KB | good | mixed_pdf | 1 | 200_1000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000261` | pdf | .pdf | 10_100KB | good | mixed_pdf | 1 | 200_1000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000283` | pdf | .pdf | 10_100KB | good | mixed_pdf | 1 | 200_1000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000335` | pdf | .pdf | 10_100KB | good | mixed_pdf | 1 | 200_1000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000360` | pdf | .pdf | 10_100KB | good | mixed_pdf | 1 | 200_1000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000388` | pdf | .pdf | 10_100KB | good | mixed_pdf | 1 | 200_1000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000405` | pdf | .pdf | 10_100KB | good | mixed_pdf | 1 | 200_1000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000423` | pdf | .pdf | 10_100KB | good | mixed_pdf | 1 | 200_1000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000424` | pdf | .pdf | 10_100KB | good | mixed_pdf | 1 | 200_1000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000450` | pdf | .pdf | 10_100KB | good | mixed_pdf | 1 | 200_1000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000459` | pdf | .pdf | 100KB_1MB | good | mixed_pdf | 3 | 1000_5000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000475` | pdf | .pdf | 10_100KB | good | mixed_pdf | 1 | 1000_5000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000484` | pdf | .pdf | 1_10MB | good | digital_pdf | 1 | 200_1000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000486` | pdf | .pdf | 100KB_1MB | good | mixed_pdf | 2 | 50_200 | yes | `likely_non_medical_or_admin` |
| `corpus_file_000507` | pdf | .pdf | 100KB_1MB | good | mixed_pdf | 3 | 1000_5000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000508` | pdf | .pdf | 100KB_1MB | good | mixed_pdf | 2 | 1000_5000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000514` | pdf | .pdf | 10_100KB | good | mixed_pdf | 1 | 200_1000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000554` | pdf | .pdf | 1_10MB | good | mixed_pdf | 5 | gt_5000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000555` | pdf | .pdf | 100KB_1MB | good | digital_pdf | 5 | gt_5000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000595` | pdf | .pdf | 100KB_1MB | good | digital_pdf | 2 | 1000_5000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000600` | pdf | .pdf | 1_10MB | usable_with_review | mixed_pdf | 2 | 1000_5000 | yes | `pdf_text_extraction_gap` |

## Privacy Safety

- uses_safe_ids_only: `True`
- raw_filenames_present_in_output: `False`
- raw_paths_present_in_output: `False`
- extracted_text_present_in_output: `False`
- ocr_text_present_in_output: `False`
- phi_present_in_output: `False`
- external_api_used: `False`

