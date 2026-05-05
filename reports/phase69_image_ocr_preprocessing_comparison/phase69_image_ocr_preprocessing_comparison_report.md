# Phase69 Image OCR Preprocessing Comparison Prototype

- Generated at: `2026-05-05T00:10:47.223905+00:00`
- Candidate count: `5`
- Meaningful improvement files: `2`
- Phase70 sandbox recommended: `False`
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
| `corpus_file_000145` | `baseline` | `very_short` | `weak` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000204` | `grayscale` | `very_short` | `weak` | `strong` | `candidate_for_phase70_sandbox` |
| `corpus_file_000205` | `baseline` | `empty` | `empty` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000206` | `threshold_binarization` | `very_short` | `weak` | `meaningful` | `candidate_for_phase70_sandbox` |
| `corpus_file_000573` | `baseline` | `very_short` | `weak` | `none` | `keep_manual_review_boundary` |

## Variant Metrics

| Safe File ID | Variant | Text Length Bucket | OCR Quality Bucket | Improvement | Next Action |
| --- | --- | --- | --- | --- | --- |
| `corpus_file_000145` | `baseline` | `very_short` | `weak` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000145` | `grayscale` | `very_short` | `weak` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000145` | `contrast_enhancement` | `very_short` | `weak` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000145` | `sharpening` | `very_short` | `weak` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000145` | `threshold_binarization` | `very_short` | `weak` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000145` | `resize_upscale` | `very_short` | `weak` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000145` | `rotate_90` | `very_short` | `poor` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000145` | `rotate_180` | `very_short` | `weak` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000145` | `rotate_270` | `very_short` | `weak` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000204` | `baseline` | `empty` | `empty` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000204` | `grayscale` | `very_short` | `weak` | `strong` | `candidate_for_phase70_sandbox` |
| `corpus_file_000204` | `contrast_enhancement` | `empty` | `empty` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000204` | `sharpening` | `very_short` | `weak` | `strong` | `candidate_for_phase70_sandbox` |
| `corpus_file_000204` | `threshold_binarization` | `short` | `weak` | `strong` | `candidate_for_phase70_sandbox` |
| `corpus_file_000204` | `resize_upscale` | `very_short` | `poor` | `strong` | `candidate_for_phase70_sandbox` |
| `corpus_file_000204` | `rotate_90` | `empty` | `empty` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000204` | `rotate_180` | `empty` | `empty` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000204` | `rotate_270` | `very_short` | `weak` | `strong` | `candidate_for_phase70_sandbox` |
| `corpus_file_000205` | `baseline` | `empty` | `empty` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000205` | `grayscale` | `empty` | `empty` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000205` | `contrast_enhancement` | `empty` | `empty` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000205` | `sharpening` | `empty` | `empty` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000205` | `threshold_binarization` | `empty` | `empty` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000205` | `resize_upscale` | `empty` | `empty` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000205` | `rotate_90` | `empty` | `empty` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000205` | `rotate_180` | `empty` | `empty` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000205` | `rotate_270` | `empty` | `empty` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000206` | `baseline` | `very_short` | `poor` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000206` | `grayscale` | `empty` | `empty` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000206` | `contrast_enhancement` | `empty` | `empty` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000206` | `sharpening` | `empty` | `empty` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000206` | `threshold_binarization` | `very_short` | `weak` | `meaningful` | `candidate_for_phase70_sandbox` |
| `corpus_file_000206` | `resize_upscale` | `empty` | `empty` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000206` | `rotate_90` | `empty` | `empty` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000206` | `rotate_180` | `empty` | `empty` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000206` | `rotate_270` | `empty` | `empty` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000573` | `baseline` | `very_short` | `weak` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000573` | `grayscale` | `very_short` | `weak` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000573` | `contrast_enhancement` | `very_short` | `weak` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000573` | `sharpening` | `very_short` | `weak` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000573` | `threshold_binarization` | `very_short` | `weak` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000573` | `resize_upscale` | `very_short` | `weak` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000573` | `rotate_90` | `very_short` | `weak` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000573` | `rotate_180` | `very_short` | `weak` | `none` | `keep_manual_review_boundary` |
| `corpus_file_000573` | `rotate_270` | `very_short` | `weak` | `none` | `keep_manual_review_boundary` |

## Safety

- Diagnostic only; production OCR routing is unchanged.
- Processed images are temporary and are not written to reports.
- OCR text is not written to public reports.
- Public reports use safe file IDs only.
