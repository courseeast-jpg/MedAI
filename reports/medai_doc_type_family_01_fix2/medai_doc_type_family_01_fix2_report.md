# MEDAI-DOC-TYPE-FAMILY-01-FIX2 Report

## Conclusion

`medai_doc_type_family_01_fix2_ready`

## Root Cause

Treatment family scoring could win from generic treatment-like cues on an imaging-style report. The classifier did not have a deterministic precedence rule for strong imaging modality plus imaging report-structure evidence.

## Conflict Resolution Rule

When Imaging report and Treatment plan evidence both appear:

- Strong imaging modality plus imaging report structure selects Imaging report if treatment evidence is generic.
- Strong evidence for both families returns an ambiguous review-bound result.
- Weak mixed generic evidence remains Unknown or ambiguous; it is not forced to Treatment plan.

## Before / After Safe Metadata

| Safe item | Before | After |
| --- | --- | --- |
| file_a | Treatment plan false positive | Imaging report with conflict resolution reason |
| file_b | Imaging report | Imaging report |
| weak_mixed_generic_case | Treatment plan | Unknown or ambiguous review-bound state |

## Behavior Changed

- Document type metadata selection now resolves imaging-versus-treatment conflicts conservatively.
- Safe diagnostic metadata includes `conflict_resolution_reason`.
- The main Run & Review card and advanced technical details remain aligned through the canonical document type metadata.

## Behavior Not Changed

- OCR routing changed: false.
- OCR engine changed: false.
- Confidence thresholds changed: false.
- Confidence scoring changed: false.
- Auto-acceptance expanded: false.
- Clinical logic changed: false.
- Imaging interpretation added: false.
- Medication parsing added: false.
- Dose parsing added: false.
- Lab value parsing added: false.
- DDI logic changed: false.
- External API behavior changed: false.

## Validation Results

- `python -m pytest tests/test_medai_doc_type_family_01_fix2_imaging_treatment_conflict.py -q` — passed, 5 passed.
- `python -m pytest tests/test_medai_doc_type_family_01_fix_runtime_imaging_propagation.py tests/test_medai_doc_type_family_01.py tests/test_medai_ui_run_review_polish_01_fix_state_consistency.py tests/test_medai_ui_run_review_polish_01.py -q` — passed, 29 passed.
- `python -m pytest tests/test_medai_ru_treatment_doc_type_01_fix2_always_emit_diagnostic.py tests/test_medai_ru_treatment_doc_type_01_fix_runtime_propagation.py tests/test_medai_ru_treatment_doc_type_01.py tests/test_medai_ru_lab_ocr_gate_02b_fix2_real_format_cue_coverage.py tests/test_medai_ru_lab_ocr_gate_02b_fix_fallback_cue_diagnostic.py tests/test_medai_ru_lab_ocr_gate_02b_local_cyrillic_fallback.py -q` — passed, 43 passed.
- `python -m pytest tests/test_medai_ru_lab_ocr_gate_02a_runtime_propagation.py tests/test_medai_ru_lab_ocr_gate_02a_shadow_marker.py tests/test_medai_ru_lab_ocr_gate_01.py tests/test_medai_ru_lab_text_vis_01.py tests/test_medai_ru_lab_extract_diag_01.py tests/test_medai_ru_doc_type_02_real_text_path_diagnostic.py tests/test_medai_ru_doc_type_01_russian_detection.py tests/test_medai_lab_fix_02_general_lab_detection.py tests/test_medai_lab_fix_01_document_type_review_reason.py tests/test_medai_ui_bugfix_02_readd_after_queue_clear.py tests/test_medai_ui_bugfix_01_clear_report_upload_idempotency.py -q` — passed, 97 passed.
- `python scripts/run_medai_ui_ops_panel_validation.py` — passed.
- `python scripts/run_medai_ui_boot_fix_validation.py` — passed.
- `python scripts/run_cka_final_mvp_release_validation.py` — passed, 12/12 cases, 693 tests passed.
- `python scripts/run_b07_term01_opt_in_integration_validation.py` — passed, 6/6 cases.
- `python scripts/run_medai_route_fix01_validation.py` — passed.
- `python -m pytest tests` — passed, 2486 passed, 4 skipped, 22 warnings.

## Privacy

- raw_ocr_text_in_public_reports: false
- raw_document_text_in_public_reports: false
- raw_filenames_private_paths_in_public_reports: false
- no_phi_in_public_reports: true
- no_secrets_in_public_reports: true

## Recommended Next Step

Repeat the same local two-document imaging smoke test and confirm both records show Imaging report, Needs review, cloud tools off, and not accepted.
