# MEDAI-DOC-TYPE-FAMILY-01-FIX

Conclusion: `medai_doc_type_family_01_fix_ready`

## Scope

This block fixes runtime propagation for document-family classification when local fallback OCR recovers text and the primary document type is still `Unknown`.

The change is metadata-only. It does not change OCR routing, OCR engine behavior, confidence scoring, thresholds, acceptance gates, or clinical interpretation.

## Root Cause

The broad document-family diagnostic was created inside fallback classification diagnostics, but the active Run & Review record did not promote that family diagnostic into the canonical result record. The runtime document-type candidate path also did not use the family diagnostic as a fallback when old classification fields remained `Unknown`.

## Runtime Fix

- Copied `document_family_classification_diagnostic` into fallback metadata and Run & Review records.
- Let the fallback metadata path use a non-Unknown family candidate only when the primary document type is missing or `Unknown`.
- Added conservative Russian imaging cue coverage for MRI/report format signals.
- Kept all affected documents review-bound.

## Safety

- Runtime behavior changed: document type metadata only.
- OCR routing changed: false
- OCR engine changed: false
- Threshold changed: false
- Auto-accept expanded: false
- Clinical logic changed: false
- Imaging interpretation added: false
- Medication parsing added: false
- Dose parsing added: false
- Lab value parsing added: false
- DDI logic changed: false
- External API changed: false
- Raw OCR text in public reports: false
- Raw document text in public reports: false
- Raw filenames/private paths in public reports: false
- Affected files remain review-bound: true

