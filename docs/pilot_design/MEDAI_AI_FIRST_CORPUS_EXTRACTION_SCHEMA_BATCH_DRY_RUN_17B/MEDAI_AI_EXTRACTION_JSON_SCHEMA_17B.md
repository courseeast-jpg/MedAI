# MEDAI AI Extraction JSON Schema 17B

This document describes the extraction schema defined in
`config/medai_ai_extraction_schema_17b.json`. 17B is dry-run only and does not call
any AI provider.

## Top-Level Fields

- `document_id` — hashed document identifier (no raw filename).
- `source_hash` — hash of the tokenized source text.
- `document_family`, `document_type` — coarse classification labels.
- `document_date_tokens` — date token placeholders only (no raw dates).
- `patient_tokens_seen`, `facility_tokens_seen` — token placeholders only.
- `extracted_labs[]`, `extracted_diagnoses[]`, `extracted_medications[]`,
  `extracted_procedures[]`, `extracted_imaging[]`, `extracted_pathology[]`,
  `extracted_vitals[]`, `extracted_notes[]` — extracted structured facts.
- `abnormal_flags[]` — abnormal indicators.
- `source_evidence[]` — tokenized evidence quotes only.
- `extraction_confidence` — model-reported confidence.
- `needs_review` — boolean; set true when uncertain.
- `extraction_warnings[]` — warnings.

## Lab Object Fields

`test_name`, `value`, `unit`, `reference_range`, `flag`, `date_token`, `specimen`,
`evidence_quote_tokenized`.

## Medication Object Fields

`medication_name`, `dose`, `route`, `frequency`, `status`, `date_token`,
`evidence_quote_tokenized`.

## Evidence Rules

- Every extracted fact must include a tokenized evidence quote.
- Evidence quotes must come from the tokenized text only.
- No raw PI is allowed anywhere in the output.
- If uncertain, set `needs_review=true`.
