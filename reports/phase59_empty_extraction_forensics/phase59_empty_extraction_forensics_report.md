# Phase 59 Empty Extraction Forensic Subset Audit

- Generated at: `2026-05-04T00:15:50.697683+00:00`
- Source Phase 57A report: `reports\phase57_full_corpus_inventory_audit\phase57_full_corpus_inventory_audit_report.json`
- Subset seed: `20260503`
- Subset size requested: `30`
- Subset size actual: `30`
- Empty extraction population: `438`
- Stratification (file_type → count): `{'pdf': 28, 'txt': 1, 'image': 1}`
- Conclusion: `forensic_subset_audited`

## Root Cause Bucket Distribution

| Bucket | Count | Percent |
| --- | ---: | ---: |
| `pdf_text_extraction_gap` | 25 | 83.33% |
| `image_only_pdf_needs_ocr` | 3 | 10.00% |
| `blank_or_near_blank` | 1 | 3.33% |
| `ocr_ran_but_low_text` | 1 | 3.33% |
| `embedded_or_portfolio_pdf` | 0 | 0.00% |
| `likely_non_medical_or_admin` | 0 | 0.00% |
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
| `corpus_file_000024` | pdf | .pdf | 100KB_1MB | good | digital_pdf | 6 | gt_5000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000085` | pdf | .pdf | 1_10MB | good | mixed_pdf | 2 | 200_1000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000096` | pdf | .pdf | 100KB_1MB | usable_with_review | scanned_pdf | 1 | 1_50 | no | `image_only_pdf_needs_ocr` |
| `corpus_file_000127` | pdf | .pdf | 1_10MB | good | digital_pdf | 2 | 1000_5000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000161` | txt | .txt | lt_1KB | unknown | text_file |  | 1_50 | no | `blank_or_near_blank` |
| `corpus_file_000206` | image | .jpg | 100KB_1MB | poor_ocr | image_ocr |  |  | no | `ocr_ran_but_low_text` |
| `corpus_file_000215` | pdf | .pdf | 100KB_1MB | good | mixed_pdf | 3 | 1000_5000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000221` | pdf | .pdf | 100KB_1MB | good | mixed_pdf | 2 | 1000_5000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000247` | pdf | .pdf | 10_100KB | good | mixed_pdf | 1 | 200_1000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000260` | pdf | .pdf | 10_100KB | good | mixed_pdf | 1 | 200_1000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000282` | pdf | .pdf | 10_100KB | good | mixed_pdf | 1 | 200_1000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000334` | pdf | .pdf | 10_100KB | good | mixed_pdf | 1 | 200_1000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000359` | pdf | .pdf | 10_100KB | good | mixed_pdf | 1 | 200_1000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000387` | pdf | .pdf | 10_100KB | good | mixed_pdf | 1 | 200_1000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000404` | pdf | .pdf | 10_100KB | good | mixed_pdf | 1 | 200_1000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000422` | pdf | .pdf | 10_100KB | good | mixed_pdf | 1 | 200_1000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000423` | pdf | .pdf | 10_100KB | good | mixed_pdf | 1 | 200_1000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000449` | pdf | .pdf | 10_100KB | good | mixed_pdf | 1 | 200_1000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000458` | pdf | .pdf | 100KB_1MB | good | mixed_pdf | 3 | 1000_5000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000473` | pdf | .pdf | 10_100KB | good | mixed_pdf | 1 | 200_1000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000483` | pdf | .pdf | 10_100KB | good | mixed_pdf | 1 | 200_1000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000485` | pdf | .pdf | 10_100KB | good | mixed_pdf | 1 | 200_1000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000506` | pdf | .pdf | 100KB_1MB | good | digital_pdf | 6 | gt_5000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000507` | pdf | .pdf | 100KB_1MB | good | mixed_pdf | 3 | 1000_5000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000512` | pdf | .pdf | 1_10MB | good | scanned_pdf | 2 | zero | no | `image_only_pdf_needs_ocr` |
| `corpus_file_000548` | pdf | .pdf | 1_10MB | good | digital_pdf | 1 | 1000_5000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000551` | pdf | .pdf | 1_10MB | good | digital_pdf | 10 | gt_5000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000591` | pdf | .pdf | 100KB_1MB | good | mixed_pdf | 2 | 1000_5000 | yes | `pdf_text_extraction_gap` |
| `corpus_file_000598` | pdf | .pdf | 1_10MB | good | scanned_pdf | 1 | zero | no | `image_only_pdf_needs_ocr` |

## Privacy Safety

- uses_safe_ids_only: `True`
- raw_filenames_present_in_output: `False`
- raw_paths_present_in_output: `False`
- extracted_text_present_in_output: `False`
- ocr_text_present_in_output: `False`
- phi_present_in_output: `False`
- external_api_used: `False`

