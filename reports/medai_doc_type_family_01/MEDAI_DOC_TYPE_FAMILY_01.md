# MEDAI-DOC-TYPE-FAMILY-01

Conclusion: `medai_doc_type_family_01_ready`

## Scope

This block adds a metadata-only document family classifier so Run & Review can identify broad document families without one-off code blocks for every new type or language.

The classifier is review-bound only. It does not parse clinical content, change OCR routing, change confidence scoring, change thresholds, or expand auto-acceptance.

## Architecture

- Added a document family registry with safe cue categories.
- Added language cue packs for English, Russian, Polish, and Albanian.
- Added safe diagnostics with cue keys, language cue groups, ambiguity status, and review-only flags.
- Preserved existing Russian lab result and treatment-plan behavior.
- Added operator-facing wording for imaging reports, clinical notes, and discharge summaries.

## Supported Families

- Lab result
- Imaging report
- Treatment plan
- Medication plan
- Clinical note
- Discharge summary
- Referral / Order
- Procedure report
- Pathology report
- Administrative / Insurance
- Unknown

## Safety

- Runtime behavior changed: document type metadata only.
- OCR routing changed: false
- Classifier clinical interpretation added: false
- Threshold changed: false
- Auto-accept expanded: false
- Clinical logic changed: false
- Medication parsing added: false
- Dose parsing added: false
- Lab value parsing added: false
- DDI logic changed: false
- External API changed: false
- Raw OCR text in public reports: false
- Raw document text in public reports: false
- Raw filenames or private paths in public reports: false
- Affected files remain review-bound: true

