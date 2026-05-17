# MEDAI-DOC-TYPE-FAMILY-01-FIX2

## Conclusion

`medai_doc_type_family_01_fix2_ready`

MEDAI-DOC-TYPE-FAMILY-01-FIX2 resolves the metadata-only conflict where a Russian imaging-style document could be classified as a Treatment plan when generic treatment-like wording appeared beside strong imaging report cues.

## Scope

- Runtime behavior changed: document type metadata only.
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
- External API changed: false.
- Affected files remain review-bound: true.

## Root Cause

The document-family classifier could treat generic treatment-style cues as enough evidence to select Treatment plan, even when stronger imaging modality and imaging report-structure cues were present. This allowed a false Treatment plan label for an imaging-style report.

## Fix Summary

- Added explicit Imaging report versus Treatment plan conflict resolution.
- Imaging report now wins when imaging modality evidence and imaging report structure are present and treatment evidence is only generic.
- Strong imaging plus strong treatment evidence remains ambiguous and review-bound instead of forcing a label.
- Weak mixed generic cues remain Unknown or ambiguous rather than becoming Treatment plan.
- Legacy Russian metadata classification now checks strong imaging evidence before treatment schedule fallback.
- Treatment family cue matching was tightened for metadata classification only. Confidence thresholds and scoring were not changed.

## Safe Before / After Metadata

| Safe item | Before | After |
| --- | --- | --- |
| file_a | candidate_family: Treatment plan; classification_block_reason: classified | candidate_family: Imaging report; conflict_resolution_reason: imaging_modality_and_report_structure_overrode_generic_treatment_cues |
| file_b | candidate_family: Imaging report; classification_block_reason: classified | candidate_family: Imaging report; classification_block_reason: classified |
| weak_mixed_generic_case | candidate_family: Treatment plan | candidate_family: Unknown or ambiguous review-bound state |

No raw OCR text, raw document text, raw filenames, private paths, PHI, or secrets are included in this report.

## Operator Result

For imaging reports, Run & Review keeps the plain operator message:

“MedAI identified this as an imaging-report style document after recovering readable Russian text locally. Imaging findings and conclusions were not interpreted or accepted. A human must review the source PDF.”

## Validation Summary

- `python -m pytest tests/test_medai_doc_type_family_01_fix2_imaging_treatment_conflict.py -q` — passed, 5 passed.
- `python -m pytest tests/test_medai_doc_type_family_01_fix_runtime_imaging_propagation.py tests/test_medai_doc_type_family_01.py tests/test_medai_ui_run_review_polish_01_fix_state_consistency.py tests/test_medai_ui_run_review_polish_01.py -q` — passed, 29 passed.
- `python -m pytest tests/test_medai_ru_treatment_doc_type_01_fix2_always_emit_diagnostic.py tests/test_medai_ru_treatment_doc_type_01_fix_runtime_propagation.py tests/test_medai_ru_treatment_doc_type_01.py tests/test_medai_ru_lab_ocr_gate_02b_fix2_real_format_cue_coverage.py tests/test_medai_ru_lab_ocr_gate_02b_fix_fallback_cue_diagnostic.py tests/test_medai_ru_lab_ocr_gate_02b_local_cyrillic_fallback.py -q` — passed, 43 passed.
- `python -m pytest tests/test_medai_ru_lab_ocr_gate_02a_runtime_propagation.py tests/test_medai_ru_lab_ocr_gate_02a_shadow_marker.py tests/test_medai_ru_lab_ocr_gate_01.py tests/test_medai_ru_lab_text_vis_01.py tests/test_medai_ru_lab_extract_diag_01.py tests/test_medai_ru_doc_type_02_real_text_path_diagnostic.py tests/test_medai_ru_doc_type_01_russian_detection.py tests/test_medai_lab_fix_02_general_lab_detection.py tests/test_medai_lab_fix_01_document_type_review_reason.py tests/test_medai_ui_bugfix_02_readd_after_queue_clear.py tests/test_medai_ui_bugfix_01_clear_report_upload_idempotency.py -q` — passed, 97 passed.
- `python scripts/run_medai_ui_ops_panel_validation.py` — passed, conclusion `medai_ui_ops_panel_ready`.
- `python scripts/run_medai_ui_boot_fix_validation.py` — passed, conclusion `medai_ui_boot_fix_startup_resilience_ready`.
- `python scripts/run_cka_final_mvp_release_validation.py` — passed, 12/12 cases, 693 tests passed.
- `python scripts/run_b07_term01_opt_in_integration_validation.py` — passed, 6/6 cases, external_api_used false.
- `python scripts/run_medai_route_fix01_validation.py` — passed, conclusion `medai_route_fix01_ready`.
- `python -m pytest tests` — passed, 2486 passed, 4 skipped, 22 warnings in 1704.38 seconds.

## Safety / Privacy

- Raw OCR text in public reports: false.
- Raw document text in public reports: false.
- Raw filenames or private paths in public reports: false.
- PHI in public reports: false.
- Secrets in public reports: false.
- Source documents staged: false.
- Runtime DB files staged: false.
- External API used: false.
