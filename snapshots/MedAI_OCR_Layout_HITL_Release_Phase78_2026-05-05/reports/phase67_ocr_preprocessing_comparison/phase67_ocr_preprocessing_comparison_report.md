# Phase67 OCR Preprocessing Comparison Prototype

- Generated at: `2026-05-04T23:37:11.787821+00:00`
- Candidate count: `2`
- Meaningful improvement files: `0`
- Phase68 sandbox recommended: `False`
- Recommended next action: `keep_manual_review_boundary`
- production_ocr_routing_changed: `False`
- production_extraction_logic_changed: `False`
- thresholds_changed: `False`
- safety_gates_changed: `False`
- external_api_used: `False`
- raw_phi_logged_in_public_reports: `False`
- conclusion: `manual_review_boundary_retained`

## Per-File Summary

| Safe File ID | Best Variant | Text Length Bucket | OCR Quality Bucket | Improvement | Next Action |
| --- | --- | --- | --- | --- | --- |
| `corpus_file_000537` | `baseline_render_150dpi` | `empty` | `empty` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000601` | `baseline_render_150dpi` | `very_short` | `weak` | `none` | `keep_manual_review_boundary` |

## Variant Metrics

| Safe File ID | Variant | Text Length Bucket | OCR Quality Bucket | Improvement | Next Action |
| --- | --- | --- | --- | --- | --- |
| `corpus_file_000537` | `baseline_render_150dpi` | `empty` | `empty` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000537` | `higher_dpi_250dpi` | `empty` | `empty` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000537` | `grayscale` | `empty` | `empty` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000537` | `contrast_sharpen` | `empty` | `empty` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000537` | `threshold_binarization` | `empty` | `empty` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000601` | `baseline_render_150dpi` | `very_short` | `weak` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000601` | `higher_dpi_250dpi` | `very_short` | `weak` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000601` | `grayscale` | `very_short` | `weak` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000601` | `contrast_sharpen` | `very_short` | `weak` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000601` | `threshold_binarization` | `very_short` | `weak` | `none` | `keep_manual_review_boundary` |

## Safety

- Diagnostic only; production OCR routing is unchanged.
- Rendered page images are temporary and are not written to reports.
- OCR text is not written to public reports.
- Public reports use safe file IDs only.
