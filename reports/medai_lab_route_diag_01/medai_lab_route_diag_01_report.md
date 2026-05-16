# MEDAI-LAB-ROUTE-DIAG-01 Report

Conclusion: `medai_lab_route_diag_01_completed`

Diagnostic type: `report_only_lab_pdf_classification_review_reason_diagnostic`

## Baseline

- Baseline smoke commit: `aac83f9c5a29`
- Baseline parked checkpoint: `882fc539`
- Privacy mode: `local_only`
- External API used: false

## Observed Smoke-Test Result

| Field | Value |
| --- | --- |
| Files analyzed | 3 |
| Accepted | 0 |
| Needs review | 3 |
| OCR / scan review | 0 |
| No text found | 0 |
| Errors | 0 |
| Observed extractors | `spacy`, `rules_based` |
| Observed document type | `unknown` |
| Observed OCR quality | `unknown` |
| Confidence range | approximately 0.45 to 0.63 |

Safe labels: `file_001`, `file_002`, `file_003`.

## Diagnosis

The review routing appears safety-preserving. The supported PDFs were not auto-accepted because confidence remained below acceptance thresholds. The primary gap is that the Run & Review path does not expose enough safe classification, OCR/text-quality, and reason metadata for the operator to understand why the files went to review.

## Root-Cause Candidates

1. Missing Run & Review metadata propagation and lab document-type integration.
2. Structured lab parser coverage or layout mismatch for the smoke-test PDFs.
3. OCR/text-quality metadata naming or propagation gap.
4. UI status mapping inconsistency between `Needs review` and `rejected`.
5. Expected safe fallback behavior from current confidence thresholds.

## Likely Primary Cause

`metadata_classification_integration_gap_combined_with_low_confidence_local_lab_extraction`

The current evidence does not indicate a ROUTE-FIX regression, an unsafe acceptance failure, or an external API issue.

## Evidence Summary

- Latest Run & Review metadata reviewed for this diagnostic did not include per-file document type, OCR/input quality band, or structured reason codes.
- The UI displays `unknown` when those fields are absent.
- Confidence around 0.45 to 0.63 is below automatic acceptance.
- The smaller `rejected` label can appear alongside a larger `Needs review` status because queue state and validation status are rendered separately.
- All uncertain results remained out of Accepted.

## Recommended Next Implementation Candidates

1. `MEDAI-LAB-FIX-01 - Lab/Urinalysis Document Type Detection + Review Reason Clarity`
2. `MEDAI-OCR-META-FIX-01 - OCR Quality Metadata Propagation`
3. `MEDAI-UI-FIX-02 - Unsupported File Type Explanation + Needs Review/Rejected Label Clarity`

## Avoid For Now

- Keep acceptance thresholds unchanged.
- Keep lab PDFs out of Accepted unless existing validation supports that outcome.
- Leave OCR/extractor/routing behavior unchanged until metadata visibility improves.
- Keep external APIs disabled.
- Keep terminology imports out of scope.

## Validation Results

| Command | Result |
| --- | --- |
| `python scripts/run_medai_ui_ops_panel_validation.py` | passed: `medai_ui_ops_panel_ready` |
| `python scripts/run_medai_ui_boot_fix_validation.py` | passed: `medai_ui_boot_fix_startup_resilience_ready` |
| `python scripts/run_cka_final_mvp_release_validation.py` | passed: 12/12 cases, 693 tests reported |
| `python scripts/run_b07_term01_opt_in_integration_validation.py` | passed: 6/6 cases |
| `python scripts/run_medai_route_fix01_validation.py` | passed: `medai_route_fix01_ready` |
| `python -m pytest tests/test_medai_ui_bugfix_02_readd_after_queue_clear.py` | passed: 6 passed |
| `python -m pytest tests/test_medai_ui_bugfix_01_clear_report_upload_idempotency.py` | passed: 6 passed |

Full pytest was not run for this report-only diagnostic.

## Safety And Privacy

- Clinical logic changed: false
- OCR/extractor changed: false
- Classifier changed: false
- Thresholds changed: false
- Safety gate changed: false
- B07 terminology changed: false
- ROUTE-FIX changed: false
- DB schema changed: false
- Command behavior changed: false
- Allowlist changed: false
- External API enabled: false
- Private files staged: false
- Source documents staged: false
- `test_input` files staged: false
- `real_validation_input` files staged: false
- Raw PHI in report: false
- Raw filenames in report: false
- Raw document text in report: false
- Private paths in report: false
- Secrets in report: false


