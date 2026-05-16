# MEDAI-LAB-ROUTE-DIAG-01

Conclusion: `medai_lab_route_diag_01_completed`

This report-only diagnostic reviewed why the first local Run & Review smoke test sent all supported lab/test-style PDFs to review with unknown document type, unknown OCR quality, and low/moderate confidence. No extraction, OCR, routing, classification, review thresholds, safety gates, UI behavior, database schema, imports, command allowlists, or external API settings were changed.

## Smoke-Test Summary

- Baseline smoke commit: `aac83f9c5a29`
- Baseline parked checkpoint: `882fc539`
- Diagnostic type: `report_only_lab_pdf_classification_review_reason_diagnostic`
- Files analyzed: 3
- Accepted: 0
- Needs review: 3
- OCR / scan review: 0
- No text found: 0
- Errors: 0
- External API used: false
- Privacy mode: `local_only`
- Observed extractors: `spacy`, `rules_based`
- Observed document type: `unknown`
- Observed OCR quality: `unknown`
- Confidence range observed: approximately 0.45 to 0.63

Safe labels used for the supported files: `file_001`, `file_002`, `file_003`.

## Ranked Root-Cause Candidates

1. Missing Run & Review metadata propagation and lab document-type integration.
   The Run & Review report path preserves selected extractor, confidence, status, and validation status, but does not surface document type, OCR/text-quality band, structured review reasons, or lower-level audit metadata. The current pipeline path also does not appear to connect the document classifier to Run & Review PDF processing, so lab/test-style PDFs can display as `unknown` even when safe lower-level metadata exists.

2. Structured lab parser coverage or layout mismatch.
   A structured lab parser exists, but the smoke-test results suggest the local extraction output was sparse for these PDFs. One file reached `spacy` with review-band confidence, while two files used `rules_based` with reject-band confidence. This points to parser/layout coverage or input normalization gaps rather than a routing-safety failure.

3. OCR/text-quality metadata naming mismatch.
   PDF processing records text-quality style metadata, but the Run & Review card expects UI fields such as OCR or input quality bands. Because those fields are not serialized into the latest run report, OCR quality is shown as `unknown`.

4. UI status mapping inconsistency.
   At least one result can show a large `Needs review` state while also showing a smaller `rejected` tag. This is confusing but safety-preserving: the file remains out of Accepted.

5. Expected safe fallback behavior from confidence thresholds.
   Observed confidence values are below automatic acceptance. Routing all three files to review is the safe outcome under the current thresholds.

## Likely Primary Cause

The safest primary diagnosis is a metadata/classification integration gap combined with low-confidence local lab extraction. The evidence does not indicate a ROUTE-FIX regression or unsafe auto-acceptance. The system behaved conservatively by routing uncertain outputs to review.

## Evidence Summary

- Latest Run & Review output stores aggregate counts and per-file extractor/confidence/status metadata only.
- Per-file document type, OCR quality, detailed review reason codes, and text-quality audit fields were not present in the public latest-run metadata reviewed for this diagnostic.
- The UI renders `unknown` when document type and OCR/input quality fields are absent.
- The UI falls back to validation status as a small reason tag when structured reason codes are absent.
- Confidence values around 0.45 to 0.63 are below automatic acceptance and explain the review/rejected states without changing thresholds.
- No supported file was auto-accepted.

## Recommended Next Blocks

1. `MEDAI-LAB-FIX-01 - Lab/Urinalysis Document Type Detection + Review Reason Clarity`
   Connect safe document-type detection and review-reason metadata into Run & Review for lab/test-style PDFs. Start by improving observability and labels without changing acceptance thresholds.

2. `MEDAI-OCR-META-FIX-01 - OCR Quality Metadata Propagation`
   Map PDF text-quality audit metadata into safe UI/report fields so the operator sees `native/readable`, `low`, `not_applicable`, or another controlled value instead of `unknown`.

3. `MEDAI-UI-FIX-02 - Unsupported File Type Explanation + Needs Review/Rejected Label Clarity`
   Clarify unsupported file-type messages and reconcile visible `Needs review` versus `rejected` status labels without changing safety behavior.

## Avoid For Now

- Keep confidence and acceptance thresholds unchanged.
- Keep lab/test-style PDFs out of Accepted unless existing validation supports that outcome.
- Leave OCR/extractor/routing behavior unchanged until safe metadata and review reasons are surfaced.
- Keep external APIs disabled.
- Keep terminology imports out of scope.
- Keep private document contents out of public reports.

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


