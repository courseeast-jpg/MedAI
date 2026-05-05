# Operator Review Checklist — Phase 71 (SAFE)

Review one file at a time. Record your answers in the private feedback file.
Do not add raw filenames, raw paths, or patient data to this document.

## Instructions

1. Open the file listed under `safe_file_id` using the private filename mapping.
2. Answer the `operator_question` for that file.
3. Choose one answer from `allowed_answers`.
4. Record your answer privately — never in this checklist.
5. Mark the item reviewed in your private feedback file.

## Allowed Answers

- `correct_accept`
- `false_accept`
- `correct_review`
- `false_review`
- `wrong_document_class`
- `unreadable_or_blank`
- `not_medical`
- `duplicate_or_bundle`
- `needs_manual_review`
- `unsure`

## Review Queue (safe IDs only)

| Priority Rank | Safe File ID | Priority Tier | Suspected Problem | Operator Question |
| --- | --- | --- | --- | --- |
| 1 | `file_001` | 1 | ocr_quality_gate_trigger | Open the file. Is it a real medical document (lab results, medication, radiology, clinical note)? Was the OCR quality gate correct to flag it? |
| 2 | `file_011` | 1 | borderline_ocr_quality | Open the file. Is it readable? Does it contain medical data? Should this have been accepted or kept in review? |
| 3 | `file_014` | 1 | flagged_needs_review | Open the file. Does it contain medical data? Was the system correct to route this to review rather than accept or empty? |
| 4 | `file_002` | 2 | unknown_document_class | Open the file. What kind of document is it? (Lab report / radiology / medication / admin / other non-medical) |
| 5 | `file_003` | 2 | unknown_document_class | Open the file. What kind of document is it? (Lab report / radiology / medication / admin / other non-medical) |
| 6 | `file_004` | 2 | unknown_document_class | Open the file. What kind of document is it? (Lab report / radiology / medication / admin / other non-medical) |
| 7 | `file_005` | 2 | unknown_document_class | Open the file. What kind of document is it? (Lab report / radiology / medication / admin / other non-medical) |
| 8 | `file_006` | 2 | unknown_document_class | Open the file. What kind of document is it? (Lab report / radiology / medication / admin / other non-medical) |
| 9 | `file_007` | 2 | unknown_document_class | Open the file. What kind of document is it? (Lab report / radiology / medication / admin / other non-medical) |
| 10 | `file_008` | 2 | unknown_document_class | Open the file. What kind of document is it? (Lab report / radiology / medication / admin / other non-medical) |
| 11 | `file_009` | 2 | unknown_document_class | Open the file. What kind of document is it? (Lab report / radiology / medication / admin / other non-medical) |
| 12 | `file_010` | 2 | unknown_document_class | Open the file. What kind of document is it? (Lab report / radiology / medication / admin / other non-medical) |
| 13 | `file_012` | 2 | unknown_document_class | Open the file. What kind of document is it? (Lab report / radiology / medication / admin / other non-medical) |
| 14 | `file_013` | 2 | unknown_document_class | Open the file. What kind of document is it? (Lab report / radiology / medication / admin / other non-medical) |
| 15 | `file_015` | 2 | unknown_document_class | Open the file. What kind of document is it? (Lab report / radiology / medication / admin / other non-medical) |

## Privacy Rules

- Do NOT write raw filenames or folder paths in any shared document.
- Do NOT write patient names, dates of birth, or other PHI anywhere outside
  a locally-secured private feedback file.
- Use safe_file_id values only when referencing files in shared context.

